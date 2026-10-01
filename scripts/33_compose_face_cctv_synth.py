"""8차 — 사람별 클래스를 가진 생성형 CCTV 인파 데이터셋 (신원 조건부 인페인팅).

scripts/32의 인파 장면에서 카메라를 향한 인물의 머리·어깨 영역을 비우고, 데이터셋
(LFW+VGGFace2) 속 **특정 인물의 신원 벡터를 조건으로 AI가 그 자리의 사람을 다시 그린다**
(SD1.5 inpainting + IP-Adapter FaceID-PlusV2, src/rareobj/faceid_inpaint.py). 얼굴을
붙이지 않으므로 경계·조명·각도·옷깃이 장면과 한 번에 생성된다. 프레임마다 여러 명이
각자 자기 클래스(사람 1명 = 클래스 1개)로 라벨링된다.

시행착오 (사용자 피드백 반영 경위)
  v1  얼굴 사진을 신체 비율로 추정한 머리 위치에 붙임 → 뒷모습 인물 뒤통수에 얼굴이
      붙는 등 매우 부자연스러움
  v2  pose 키포인트로 정면 인물만 골라 눈·코 정렬 + img2img 경계 섞기 → 본질이 여전히
      "얼굴 붙이기"라 폐기
  v3  (현재) 신원 조건부 인페인팅. 첫 시험에서 머리만 선명하고 주변이 뭉개져 튀었다 —
      원인은 작은 인물(얼굴 ~11px) 주변을 10배 확대해 그린 것. 장면을 가까운 군중
      구도로 바꾸고(얼굴 중앙값 ~19px), 프레임을 x2(ESRGAN x4 후 축소, 선명)로 렌더링,
      그 해상도에서 얼굴 창만 512로 확대해 그린 뒤 다시 프레임 해상도로 축소해 넣는다
      → 카메라 해상도가 프레임 전체에서 균일. 또 생성 장면의 걷는 군중은 모션블러가
      있어 그린 사람만 선명하면 튀므로, 그린 영역의 선명도를 주변에 맞춰 흐린다.

얼굴 한 명 넣는 순서
  1. 창    슬롯(눈·코 키포인트) 중심 3.8x 얼굴 크기 정사각 창, 512로 확대
  2. 마스크 머리 타원 + 어깨 사다리꼴 (그 인물의 머리~상반신을 새로 그림)
  3. 인페인팅  찾을 사람의 정면 사진 최대 4장 → ArcFace 평균 신원 벡터 + CLIP 얼굴 특징
  4. 축소·블러 일치  프레임 해상도로 축소, 마스크 안 선명도를 마스크 둘레 맥락에 맞춤
  5. 신원 검증  창에서 얼굴 재검출 → CASIA 임베딩으로 원본 사진 평균과 코사인(id_cos).
     ID_COS_MIN 미만이면 되돌림(그 자리는 원래 생성 인물 = 모르는 사람)
그 다음 프레임 전체에 CCTV 열화를 걸고, 라벨 박스로 얼굴을 다시 잘라 학습 샘플로 쓴다.

split
  train  32 train 장면 x train 신원 480명 (클래스 0..479) → 학습
  test   32 test 장면 x eval_vgg 신원 60명 (학습에 없는 사람) → 인파 속 특정인 찾기 평가.
         바꾸지 않은 정면 인물은 "모르는 사람"(오인 측정용)으로 남는다

출력
  data/dataset/faces_v1/cctv_frames/{train,test}/images/*.jpg
  data/dataset/faces_v1/cctv_frames/{train,test}/labels/*.txt   YOLO 형식 (cls=사람 클래스)
  data/dataset/faces_v1/cctv_frames/{train,test}/frames.jsonl   박스·신원·id_cos·미변경 인물
  data/dataset/faces_v1/synth_cctv/<gid>/*.jpg  + synth_cctv_manifest.csv  (학습용 얼굴 크롭)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rareobj import degrade, paths, restore  # noqa: E402
from rareobj.faceid_inpaint import FaceIDInpainter  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

ROOT = paths.DATASET / "faces_v1"
GEN = paths.INTERIM / "generated"
SCENES = GEN / "face_cctv_scenes"
SCENE_MANIFEST = GEN / "face_cctv_scenes_manifest.jsonl"
FRAMES = ROOT / "cctv_frames"
ESRGAN_CKPT = paths.CHECKPOINTS / "realesrgan" / "RealESRGAN_x4plus.pth"
OUT_SIZE, MARGIN = 160, 14
F_RENDER = 2.0                # 프레임 = 생성 장면 x2 (ESRGAN x4 → 축소)
FACE_PER_EYE = 2.5            # 얼굴 한 변 ≈ 눈 사이 거리 x 2.5
WIN_MULT = 3.8                # 인페인팅 창 = 얼굴 x 3.8
SIDE, STEPS = 512, 20
N_TRY = 1                     # 재시도: 20장면 시험에서 2회차가 8명만 더 살리고 시간은 +40% → 끔
SHORTCUT_SCALE = 1.5          # FaceID-PlusV2 의 CLIP 얼굴 특징 반영 비율 (기본 1.0 — 신원 강화)
ID_COS_MIN = 0.40             # 시험에서 같은 사람 0.5~0.62 / 모르는 사람 0.01~0.07
MIN_CTX_LAP = 10.0            # 창 맥락(프레임 해상도) 라플라시안 분산 하한 — 너무 뭉개진 장면 제외
N_SRC = 4
FRAME_RANGES = {"blur": (1, 4), "down": (0.65, 1.0), "sigma": (1.0, 5.0),
                "poisson": (80.0, 240.0), "jpeg": (35, 80)}
COMPS_PER_SCENE = {"train": 2, "test": 2}
TEST_REPLACE_FRAC = 0.6
PROMPT = {False: "photo of a person walking in a crowd, face looking at the camera, cctv security "
                 "camera footage, realistic, natural lighting",
          True: "photo of a person walking in a crowd at night, face looking at the camera, night "
                "time cctv security camera footage, artificial street lighting, realistic"}
NEGATIVE = ("cartoon, painting, illustration, deformed face, distorted, extra eyes, mask, "
            "watermark, text, monochrome")


class Models:
    def __init__(self):
        from facenet_pytorch import MTCNN, InceptionResnetV1

        self.fi = FaceIDInpainter()
        self.esrgan = restore.load_esrgan(ESRGAN_CKPT).half()
        self.mtcnn = MTCNN(keep_all=True, device="cuda")
        self.casia = InceptionResnetV1(pretrained="casia-webface").eval().cuda()

    @torch.no_grad()
    def upscale4(self, bgr, tile=256, pad=12):
        H, W = bgr.shape[:2]
        x = torch.from_numpy(bgr[:, :, ::-1].astype(np.float32) / 255).permute(2, 0, 1)[None].cuda().half()
        out = torch.zeros(1, 3, H * 4, W * 4, device="cuda", dtype=torch.half)
        for y0 in range(0, H, tile):
            for x0 in range(0, W, tile):
                ya, xa = max(y0 - pad, 0), max(x0 - pad, 0)
                yb, xb = min(y0 + tile + pad, H), min(x0 + tile + pad, W)
                o = self.esrgan(x[:, :, ya:yb, xa:xb])
                oy, ox = (y0 - ya) * 4, (x0 - xa) * 4
                th, tw = min(tile, H - y0) * 4, min(tile, W - x0) * 4
                out[:, :, y0 * 4:y0 * 4 + th, x0 * 4:x0 * 4 + tw] = o[:, :, oy:oy + th, ox:ox + tw]
        rgb = out[0].clamp(0, 1).permute(1, 2, 0).float().cpu().numpy()
        return np.ascontiguousarray((rgb[:, :, ::-1] * 255).round().astype(np.uint8))

    @torch.no_grad()
    def casia_embed(self, crops):
        x = torch.stack([torch.from_numpy((c[:, :, ::-1].astype(np.float32) - 127.5) / 128).permute(2, 0, 1)
                         for c in crops]).cuda()
        return F.normalize(self.casia(x)).cpu().numpy()

    def detect(self, bgr, landmarks=False):
        return self.mtcnn.detect(Image.fromarray(bgr[:, :, ::-1]), landmarks=landmarks)


class IdentityBank:
    """신원별 생성 조건(ArcFace 평균 + CLIP 얼굴) 과 검증용 CASIA 템플릿을 캐시."""

    def __init__(self, models, pool, rng):
        self.m, self.pool, self.rng, self.cache = models, pool, rng, {}

    def get(self, gid):
        if gid in self.cache:
            return self.cache[gid]
        faces, paths_ = [], []
        cands = list(self.pool[gid])
        self.rng.shuffle(cands)
        for r in cands[:60]:
            im = cv2.imread(str(ROOT / r["path"]))
            _, _, lm = self.m.detect(im, landmarks=True)
            if lm is None:
                continue
            l = lm[0]
            ed = float(np.hypot(*(l[0] - l[1])))
            mid = (l[0] + l[1]) / 2
            # 정면 사진만 (코가 두 눈 중앙, 눈높이 수평, 충분한 해상도)
            if abs(l[2][0] - mid[0]) < 0.12 * ed and abs(l[0][1] - l[1][1]) < 0.15 * ed and ed > 40:
                faces.append((im, l))
                paths_.append(r["path"])
            if len(faces) >= N_SRC:
                break
        if not faces:
            self.cache[gid] = None
            return None
        idv, clip_face = self.m.fi.identity_embeds(faces)
        t = self.m.casia_embed([f[0] for f in faces]).mean(0)
        self.cache[gid] = (idv, clip_face, t / np.linalg.norm(t), paths_)
        return self.cache[gid]


def lap_var(bgr, mask=None):
    g = cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
    return float(g[mask > 0.5].var()) if mask is not None else float(g.var())


def match_blur(gen, ctx_ref, inner, ring):
    """gen 의 마스크 안 선명도가 둘레 맥락보다 높으면 같아질 때까지 가우시안 블러."""
    target = lap_var(ctx_ref, ring)
    if lap_var(gen, inner) <= target:
        return gen, 0.0
    lo, hi = 0.0, 4.0
    for _ in range(7):
        mid = (lo + hi) / 2
        b = cv2.GaussianBlur(gen, (0, 0), mid)
        lo, hi = (mid, hi) if lap_var(b, inner) > target else (lo, mid)
    return cv2.GaussianBlur(gen, (0, 0), hi), hi


def make_mask(side, fs):
    c = side / 2
    m = np.zeros((side, side), np.float32)
    cv2.ellipse(m, (int(c), int(c - 0.12 * fs)), (int(0.72 * fs), int(0.95 * fs)), 0, 0, 360, 1.0, -1)
    # 어깨 사다리꼴은 창 아래 끝(c + 1.9fs)보다 0.4fs 위에서 끝낸다 — 처음엔 창 끝까지
    # 닿게 했더니 창 밖 원본과 맞닿는 곳에 가로 이음매가 보였다
    bot = c + 1.5 * fs
    pts = np.int32([[c - 0.45 * fs, c + 0.55 * fs], [c + 0.45 * fs, c + 0.55 * fs],
                    [c + 1.15 * fs, bot], [c - 1.15 * fs, bot]])
    cv2.fillPoly(m, [pts], 1.0)
    return m


def border_taper(n, frac=0.12):
    """창 테두리로 갈수록 0 — 창 경계에서 원본과 이음매가 생기지 않게 최종 블렌딩에 곱한다."""
    r = np.minimum(np.arange(n), np.arange(n)[::-1]).astype(np.float32) / max(1, frac * n)
    t = np.clip(r, 0, 1)
    return np.minimum(t[:, None], t[None, :])


def crop_box(frame, box):
    H, W = frame.shape[:2]
    bw, bh = box[2] - box[0], box[3] - box[1]
    mx, my = MARGIN * bw / (OUT_SIZE - MARGIN), MARGIN * bh / (OUT_SIZE - MARGIN)
    x0, y0 = int(max(box[0] - mx / 2, 0)), int(max(box[1] - my / 2, 0))
    x1, y1 = int(min(box[2] + mx / 2, W)), int(min(box[3] + my / 2, H))
    return cv2.resize(frame[y0:y1, x0:x1], (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_CUBIC)


def get_window(canvas, x0, y0, n):
    """프레임 밖으로 나가는 창은 반사 패딩으로 채운다. 반환: (창, 프레임 안 유효 영역 슬라이스들)."""
    H, W = canvas.shape[:2]
    ax0, ay0, ax1, ay1 = max(x0, 0), max(y0, 0), min(x0 + n, W), min(y0 + n, H)
    sub = canvas[ay0:ay1, ax0:ax1]
    win = cv2.copyMakeBorder(sub, ay0 - y0, (y0 + n) - ay1, ax0 - x0, (x0 + n) - ax1, cv2.BORDER_REFLECT101)
    return win, (slice(ay0, ay1), slice(ax0, ax1)), (slice(ay0 - y0, ay1 - y0), slice(ax0 - x0, ax1 - x0))


def insert_person(models, canvas, slot, ident, night, rng, stats):
    """canvas 제자리 수정. 성공 시 (얼굴 박스, id_cos), 실패 시 None (canvas 원상태).

    신원 검증에 실패하면 시드를 바꿔 N_TRY 번까지 다시 그린다 — 1차 전체 실행에서
    탈락 사유 1위가 신원 불일치(45%)였다."""
    idv, clip_face, casia_t, _ = ident
    fs0 = FACE_PER_EYE * slot["eye_dist"] * F_RENDER
    cx = (slot["leye"][0] + slot["reye"][0]) / 2 * F_RENDER
    cy = (slot["leye"][1] + slot["reye"][1]) / 2 * F_RENDER + 0.15 * fs0
    n = int(WIN_MULT * fs0)
    H, W = canvas.shape[:2]
    x0, y0 = int(round(cx - n / 2)), int(round(cy - n / 2))
    # 1차 실행에서는 창이 프레임 밖으로 나가면 버렸다(11%) — 반사 패딩으로 살리고,
    # 얼굴 자체가 프레임 밖에 걸리는 경우만 아래에서 버린다
    orig, dst, src = get_window(canvas, x0, y0, n)
    mask_n = cv2.resize(make_mask(SIDE, fs0 * SIDE / n), (n, n), interpolation=cv2.INTER_AREA)
    if lap_var(orig, 1 - mask_n) < MIN_CTX_LAP:
        stats["ctx_too_blurry"] += 1
        return None
    win = cv2.resize(orig, (SIDE, SIDE), interpolation=cv2.INTER_CUBIC)
    mask = make_mask(SIDE, fs0 * SIDE / n)
    k = max(3, (n // 20) | 1)
    inner = cv2.erode(mask_n, np.ones((k, k), np.uint8))
    ring = cv2.dilate(mask_n, np.ones((k * 3, k * 3), np.uint8)) - cv2.dilate(mask_n, np.ones((k, k), np.uint8))
    m2 = cv2.GaussianBlur(cv2.dilate(mask_n, np.ones((k, k), np.uint8)), (k * 2 + 1, k * 2 + 1), 0)
    m2 = (m2 * border_taper(n))[..., None]
    for t in range(N_TRY):
        out = models.fi.inpaint(win, mask, idv, clip_face, PROMPT[night], NEGATIVE,
                                seed=rng.randint(0, 2**31 - 1), steps=STEPS, shortcut_scale=SHORTCUT_SCALE)
        gen = cv2.resize(out, (n, n), interpolation=cv2.INTER_AREA)       # 프레임 해상도로
        gen, sigma = match_blur(gen, orig, inner, ring)
        comp = (gen * m2 + orig * (1 - m2)).astype(np.uint8)
        # 그려진 얼굴 재검출 → 라벨 박스 + 신원 검증
        boxes, probs = models.detect(cv2.resize(comp, (SIDE, SIDE), interpolation=cv2.INTER_CUBIC))
        if boxes is None:
            stats["no_face_after"] += 1
            continue
        s = n / SIDE
        c = SIDE / 2
        j = int(np.argmin([abs((b[0] + b[2]) / 2 - c) + abs((b[1] + b[3]) / 2 - (c - 0.15 * fs0 / s))
                           for b in boxes]))
        if probs[j] < 0.9:
            stats["no_face_after"] += 1
            continue
        b = boxes[j] * s
        box = [float(b[0] + x0), float(b[1] + y0), float(b[2] + x0), float(b[3] + y0)]
        if box[0] < 0 or box[1] < 0 or box[2] > W or box[3] > H:
            stats["face_off_frame"] += 1
            return None
        before = canvas[dst].copy()
        canvas[dst] = comp[src]
        cos = float(models.casia_embed([crop_box(canvas, box)])[0] @ casia_t)
        stats["cos_all"].append(cos)
        if cos >= ID_COS_MIN:
            stats["blur_sigma"].append(sigma)
            stats[f"pass_try{t + 1}"] += 1
            return box, cos
        canvas[dst] = before
        stats["id_gate_reject"] += 1
    return None


def frame_degrade(frame, rng):
    p = degrade.DegradeParams(
        motion_blur_len=rng.randint(*FRAME_RANGES["blur"]), motion_blur_angle=rng.uniform(0, 180),
        downscale_factor=rng.uniform(*FRAME_RANGES["down"]), noise_sigma=rng.uniform(*FRAME_RANGES["sigma"]),
        poisson_scale=rng.uniform(*FRAME_RANGES["poisson"]), jpeg_quality=rng.randint(*FRAME_RANGES["jpeg"]))
    return degrade.apply_pipeline(frame, p), p


def unknown_box(slot):
    ce = np.array([(slot["leye"][0] + slot["reye"][0]) / 2, (slot["leye"][1] + slot["reye"][1]) / 2]) * F_RENDER
    sd = FACE_PER_EYE * slot["eye_dist"] * F_RENDER
    return [round(float(ce[0] - sd / 2), 1), round(float(ce[1] - sd * 0.4), 1),
            round(float(ce[0] + sd / 2), 1), round(float(ce[1] + sd * 0.6), 1)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-scenes", type=int, default=0, help="디버그용 split별 장면 수 제한")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--seed", type=int, default=20261001)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    models = Models()

    rows = list(csv.DictReader(open(ROOT / "manifest.csv", encoding="utf-8")))
    proto = json.loads((ROOT / "eval_protocol.json").read_text(encoding="utf-8"))
    enrolled = set(proto["gallery_enroll"])
    pool = {"train": defaultdict(list), "test": defaultdict(list)}
    for r in rows:
        if r["role"] == "train":
            pool["train"][int(r["gid"])].append(r)
        # test 인물의 생성 조건 사진은 갤러리 등록 사진과 겹치지 않게
        elif r["role"] == "eval_vgg" and r["path"] not in enrolled:
            pool["test"][int(r["gid"])].append(r)
    gid2cls = {int(r["gid"]): int(r["cls"]) for r in rows if r["role"] == "train"}
    banks = {s: IdentityBank(models, pool[s], rng) for s in pool}

    scenes = [json.loads(l) for l in open(SCENE_MANIFEST, encoding="utf-8")]
    crop_rows = []
    stats = defaultdict(int)
    stats["cos_all"], stats["blur_sigma"] = [], []
    use_count = {s: defaultdict(int) for s in pool}
    t0 = time.time()
    for split in args.splits:
        sc = [s for s in scenes if s["split"] == split]
        if args.max_scenes:
            sc = sc[:args.max_scenes]
        d = FRAMES / split
        (d / "images").mkdir(parents=True, exist_ok=True)
        (d / "labels").mkdir(parents=True, exist_ok=True)
        fj = open(d / "frames.jsonl", "w", encoding="utf-8")
        for si, scene in enumerate(sc):
            base = cv2.imread(str(SCENES / scene["file"]))
            canvas0 = cv2.resize(models.upscale4(base), (int(base.shape[1] * F_RENDER), int(base.shape[0] * F_RENDER)),
                                 interpolation=cv2.INTER_AREA)
            for comp in range(COMPS_PER_SCENE[split]):
                canvas = canvas0.copy()
                slots = list(scene["slots"])
                rng.shuffle(slots)
                n_rep = len(slots) if split == "train" else max(1, round(len(slots) * TEST_REPLACE_FRAC))
                ids = sorted(pool[split], key=lambda g: (use_count[split][g], rng.random()))[:n_rep]
                faces, unknown = [], [unknown_box(s) for s in slots[n_rep:]]
                for slot, gid in zip(slots, ids):
                    ident = banks[split].get(gid)
                    res = None if ident is None else insert_person(
                        models, canvas, slot, ident, scene["night"], rng, stats)
                    if res is None:
                        unknown.append(unknown_box(slot))
                        continue
                    box, cos = res
                    use_count[split][gid] += 1
                    faces.append({"gid": gid, "cls": gid2cls.get(gid, -1), "src": ident[3],
                                  "box": [round(v, 1) for v in box], "id_cos": round(cos, 3)})
                if not faces:
                    stats["frame_no_face"] += 1
                    continue
                frame, p = frame_degrade(canvas, rng)
                name = f"{Path(scene['file']).stem}_{comp}"
                cv2.imwrite(str(d / "images" / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                Hh, Ww = frame.shape[:2]
                with open(d / "labels" / f"{name}.txt", "w", encoding="utf-8") as lf:
                    for fc in faces:
                        b = fc["box"]
                        cls = fc["cls"] if split == "train" else fc["gid"]
                        lf.write(f"{cls} {(b[0] + b[2]) / 2 / Ww:.6f} {(b[1] + b[3]) / 2 / Hh:.6f} "
                                 f"{(b[2] - b[0]) / Ww:.6f} {(b[3] - b[1]) / Hh:.6f}\n")
                fj.write(json.dumps({"image": f"images/{name}.jpg", "scene": scene["file"], "size": [Ww, Hh],
                                     "night": scene["night"], "faces": faces, "unknown_frontal": unknown,
                                     "degrade": p.to_dict()}, ensure_ascii=False) + "\n")
                fj.flush()
                stats[f"{split}_frames"] += 1
                stats[f"{split}_faces"] += len(faces)
                if split == "train":
                    for k, fc in enumerate(faces):
                        rel = Path("synth_cctv") / f"{fc['gid']:05d}" / f"{name}_{k}.jpg"
                        (ROOT / rel).parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(ROOT / rel), crop_box(frame, fc["box"]), [cv2.IMWRITE_JPEG_QUALITY, 95])
                        crop_rows.append({"path": rel.as_posix(), "gid": fc["gid"], "cls": fc["cls"],
                                          "frame": f"cctv_frames/train/images/{name}.jpg",
                                          "face_px": round(fc["box"][2] - fc["box"][0], 1),
                                          "id_cos": fc["id_cos"], "night": int(scene["night"])})
            if si % 20 == 0:
                c = stats["cos_all"]
                summ = {k: v for k, v in stats.items() if not isinstance(v, list)}
                print(f"  [{split}] scene {si}/{len(sc)}  {summ}  id_cos med "
                      f"{np.median(c) if c else float('nan'):.3f}  {time.time() - t0:.0f}s", flush=True)
        fj.close()

    if crop_rows:
        with open(ROOT / "synth_cctv_manifest.csv", "w", newline="", encoding="utf-8") as fo:
            wr = csv.DictWriter(fo, fieldnames=list(crop_rows[0]))
            wr.writeheader()
            wr.writerows(crop_rows)
    c = np.array(stats["cos_all"])
    print(f"[done] {({k: v for k, v in stats.items() if not isinstance(v, list)})}")
    if len(c):
        print(f"  id_cos (검증 전 전체) q10 {np.quantile(c, .1):.3f}  med {np.median(c):.3f}  "
              f"q90 {np.quantile(c, .9):.3f}  통과율 {(c >= ID_COS_MIN).mean():.2%}")
    print(f"  blur-match sigma med {np.median(stats['blur_sigma']) if stats['blur_sigma'] else 0:.2f}")
    print(f"  train crops {len(crop_rows)}  classes covered {len({r['cls'] for r in crop_rows})}/480")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
