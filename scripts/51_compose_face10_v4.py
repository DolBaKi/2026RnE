"""v4 — 10명 타깃 + 모르는 사람 CCTV 인파 프레임 (장면·인물 모두 RealVisXL).

v3(scripts/41 + 41b)와 다른 점 (2026-10-29, 사용자: "흰색 경계만 학습하는 양상", "포토샵처럼 제대로")
  1. 장면(scripts/50)과 인물을 같은 RealVisXL 이 그린다 — v3 는 SDXL-Turbo 장면 + SD1.5 머리라
     질감·초점이 달랐다. (src/rareobj/faceid_inpaint_xl.py)
  2. 정면 얼굴은 타깃이든 아니든 큰 것부터 MAX_RENDER 개를 전부 같은 방식으로 다시 그린다.
     타깃이 아닌 자리는 모르는 사람(성별 맞춤)으로 — "손댄 얼굴 vs 손 안 댄 얼굴" 차이 자체를 없앤다.
  3. 마스크는 머리+목만, 경계는 라플라시안 피라미드 블렌딩. 그린 얼굴의 색·밝기·대비·선명도를
     그 자리에 원래 있던 얼굴에 맞춘다 (harmonize — 등록 사진의 화장·스튜디오 조명이 따라오고,
     고해상도로 그려 축소한 얼굴이 주변보다 선명해 scripts/52 판별 검사에 걸렸다).
  4. FaceID 강도는 샘플마다 shortcut 0.7~1.0, ip 0.65~0.85 — 시험에서 v3 의 1.5/1.0 은 등록 사진의
     표정·조명까지 복사해 캐리커처처럼 보였고, 낮춰도 신원 코사인 0.6~0.75 를 유지했다.
  5. 삽입이 끝난 프레임 전체를 같은 모델로 원해상도에서 낮은 강도(0.12)로 한 번 더 그린다 — 넣은 얼굴과
     원래 얼굴이 같은 렌더링을 거쳐 질감·주파수 특성이 가까워진다. 30장면 파일럿(정면 대조군 판별 AUC,
     CNN / 신원 코사인 중앙값): 0 → 0.90 / 0.64, 0.12 → 0.84 / 0.42, 0.25 → 0.79 / 0.30. v3 는 0.93.
     재생성 후 신원을 다시 재고(id_cos_final) 미달 얼굴은 학습 크롭에서 뺀다.
  6. v3 의 SDXL 저주파 톤 옮기기(41b)는 뺐다 — 얼굴만 11~25 어둡게 하고 주변을 밝혀 흰 테두리를
     만든 원인. 렌즈·OSD·CCTV 열화는 프레임 전체에 균일하게 건다.

  train  scripts/50 train 장면 / 타깃 사진은 faces10 train 분할, 모르는 사람은 VGGFace2 나머지 470명
  test   scripts/50 test 장면 / 타깃 사진은 test_hq 분할, 모르는 사람은 eval_vgg 60명

출력 (scripts/35·42·43 이 읽는 v3 와 같은 형식)
  <out>/cctv_frames/{train,test}/images, labels(YOLO), frames.jsonl
  <out>/synth_cctv/<cls>/*.jpg + <out>/synth_manifest.csv  (train 얼굴 크롭)
  <out> 기본값 = data/dataset/faces10 (기존 v3 출력은 *_v3 로 옮겨 두고 실행)
"""

from __future__ import annotations

import argparse
import csv
import importlib
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rareobj import cctv_style as cs, genmodels, paths  # noqa: E402
from rareobj.faceid_inpaint_xl import FaceIDInpainterXL, pyramid_blend  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

c33 = importlib.import_module("33_compose_face_cctv_synth")      # get_window, border_taper, crop_box, frame_degrade
s50 = importlib.import_module("50_generate_face_scenes_v4")
F10 = paths.DATASET / "faces10"
V1 = paths.DATASET / "faces_v1"
UGENDER_CACHE = F10 / "unknown_gender.json"
GLASSES_CACHE = F10 / "identity_glasses.json"
MAX_FAILS = 5                 # 성공 없이 이만큼 연속 실패한 신원은 건너뛴다
UNKNOWN = 10
MAX_TARGETS = 2
MAX_RENDER = 5                # 프레임당 다시 그릴 정면 얼굴 수 (큰 것부터) — 삽입 1건 ~12초
MIN_RENDER_FACE = 20.0        # 얼굴 한 변(px) 이보다 작은 슬롯은 그리지 않는다 (원본도 이 크기면 구분 불가)
SIDE, STEPS = 768, 20
WIN_MULT = 4.0
SHORTCUT = (0.7, 1.0)
IP_SCALE = (0.65, 0.85)
ID_COS_MIN = 0.40             # v3 와 같은 신원 검증 기준 (CASIA, 생성 조건의 ArcFace 와 분리)
K1, VIGNETTE = (0.04, 0.12), (0.15, 0.35)
N_SRC = 4
PROMPT = {False: "photo of a {g} walking in a crowd, face looking at the camera, neutral expression, mouth closed, "
                 "cctv security camera footage, overcast flat lighting, deep depth of field, realistic photo",
          True: "photo of a {g} walking in a crowd at night, face looking at the camera, neutral expression, mouth "
                "closed, night time cctv security camera footage, artificial street lighting, deep depth of field, "
                "realistic photo"}
NEGATIVE = ("open mouth, talking, shouting, grimace, teeth, bokeh, blurry face, depth of field, studio lighting, "
            "makeup, cartoon, painting, deformed face, distorted, extra eyes, mask, watermark, text")
GENDER_WORD = {"M": "man", "F": "woman"}


def head_mask(side, fs):
    """머리(머리카락 포함) 타원 + 목. 어깨·가슴은 원본 그대로 — 사다리꼴 어깨 마스크는 가슴에 경계가 보였다."""
    c = side / 2
    m = np.zeros((side, side), np.float32)
    cv2.ellipse(m, (int(c), int(c - 0.12 * fs)), (int(0.70 * fs), int(0.92 * fs)), 0, 0, 360, 1.0, -1)
    cv2.ellipse(m, (int(c), int(c + 0.70 * fs)), (int(0.38 * fs), int(0.40 * fs)), 0, 0, 360, 1.0, -1)
    return m


def slot_geom(s):
    fs = s50.FACE_PER_EYE * s["eye_dist"]
    return fs, (s["leye"][0] + s["reye"][0]) / 2, (s["leye"][1] + s["reye"][1]) / 2


def slot_box(s):
    """그리지 않은 정면 얼굴의 근사 박스 (eye 중심 기준)."""
    fs, cx, cy = slot_geom(s)
    return [round(cx - fs / 2, 1), round(cy - fs * 0.4, 1), round(cx + fs / 2, 1), round(cy + fs * 0.6, 1)]


def scene_face_ref(img, slots):
    """장면 속 얼굴들의 Lab 중앙값과 밝기 표준편차 중앙값 — 그린 얼굴을 이 조명·대비에 맞춘다."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    meds, stds = [], []
    for s in slots:
        fs, cx, cy = slot_geom(s)
        p = lab[max(0, int(cy - .1 * fs)):int(cy + .45 * fs), max(0, int(cx - .25 * fs)):int(cx + .25 * fs)]
        q = lab[max(0, int(cy - .3 * fs)):int(cy + .45 * fs), max(0, int(cx - .3 * fs)):int(cx + .3 * fs), 0]
        if p.size and q.size:
            meds.append(np.median(p.reshape(-1, 3), 0))
            stds.append(q.std())
    return (np.median(meds, 0), float(np.median(stds))) if meds else (None, None)


def face_core(n, fs):
    h = n / 2     # 창 중심 = 눈 중심 + 0.15fs → 얼굴 피부 영역은 대략 중심 위아래
    return slice(int(h - .35 * fs), int(h + .25 * fs)), slice(int(h - .25 * fs), int(h + .25 * fs))


def harmonize(gen, orig, alpha, fs, ref_lab):
    """그린 얼굴의 저수준 통계를 '바로 그 자리에 원래 있던 얼굴'(같은 위치·크기·조명)에 맞춘다.

    파일럿 판별 검사(scripts/52, 12장면): 넣은 얼굴이 원래 얼굴보다 붉은기 분산(a std, AUC 0.96)·
    밝기(L 102 vs 84)·선명도(라플라시안 8.2 vs 4.9)가 커서 수작업 특징만으로 AUC 0.998 로 갈렸다.
    처음엔 장면 얼굴들의 중앙값 색으로만 옮겼는데, 그 이동을 머리카락까지 똑같이 걸어 검은 머리가
    청록으로 물들기도 했다 → 색(a,b) 이동은 피부색에 가까운 픽셀에만 건다.
      L     평균은 원래 얼굴 쪽으로 70%, 대비(std)는 원래 얼굴 기준 0.5~1.1배
      a,b   평균은 원래 얼굴·장면 얼굴 중앙값의 절반씩 쪽으로, 분산은 원래 얼굴 기준 (줄이기 위주)
    """
    n = gen.shape[0]
    ys, xs = face_core(n, fs)
    lab = cv2.cvtColor(gen, cv2.COLOR_BGR2LAB).astype(np.float32)
    olab = cv2.cvtColor(orig, cv2.COLOR_BGR2LAB).astype(np.float32)
    g, o = lab[ys, xs].reshape(-1, 3), olab[ys, xs].reshape(-1, 3)
    if len(g) < 4:
        return gen
    mu_g, sd_g, mu_o, sd_o = g.mean(0), g.std(0) + 1e-3, o.mean(0), o.std(0) + 1e-3
    mu_t = mu_o.copy()
    if ref_lab is not None:
        mu_t[1:] = 0.5 * mu_o[1:] + 0.5 * ref_lab[1:]
    k = np.clip(sd_o / sd_g, 0.5, 1.1)
    w_mean = np.array([0.7, 1.0, 1.0], np.float32)
    skin_ab = np.median(g[:, 1:], 0)
    d2 = ((lab[..., 1:] - skin_ab) ** 2).sum(-1)
    w_skin = np.exp(-d2 / (2 * 6.0 ** 2))                      # 피부색에서 먼 픽셀(머리카락·옷깃)은 색 보정 약하게
    out = lab.copy()
    for ch in range(3):
        tgt = (lab[..., ch] - mu_g[ch]) * k[ch] + mu_g[ch] + (mu_t[ch] - mu_g[ch]) * w_mean[ch]
        wt = alpha if ch == 0 else alpha * w_skin
        out[..., ch] = lab[..., ch] + (tgt - lab[..., ch]) * wt
    gen = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    # 선명도: 그린 얼굴(고해상도로 그려 축소 → 디테일 과다)을 원래 얼굴 선명도까지만 흐리게
    gy = cv2.Laplacian(cv2.cvtColor(gen, cv2.COLOR_BGR2GRAY), cv2.CV_64F)[ys, xs].var()
    oy = cv2.Laplacian(cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY), cv2.CV_64F)[ys, xs].var()
    if gy > oy:
        lo, hi = 0.0, 3.0
        for _ in range(7):
            mid = (lo + hi) / 2
            v = cv2.Laplacian(cv2.cvtColor(cv2.GaussianBlur(gen, (0, 0), mid), cv2.COLOR_BGR2GRAY), cv2.CV_64F)[ys, xs].var()
            lo, hi = (mid, hi) if v > oy else (lo, mid)
        blurred = cv2.GaussianBlur(gen, (0, 0), hi)
        gen = (blurred * alpha[..., None] + gen * (1 - alpha[..., None])).astype(np.uint8)
    return gen


class Models:
    def __init__(self):
        from facenet_pytorch import MTCNN, InceptionResnetV1

        self.fi = FaceIDInpainterXL()
        self.mtcnn = MTCNN(keep_all=True, device="cuda")
        self.casia = InceptionResnetV1(pretrained="casia-webface").eval().cuda()

    @torch.no_grad()
    def casia_embed(self, crops):
        x = torch.stack([torch.from_numpy((c[:, :, ::-1].astype(np.float32) - 127.5) / 128).permute(2, 0, 1)
                         for c in crops]).cuda()
        return F.normalize(self.casia(x)).cpu().numpy()

    def detect(self, bgr, landmarks=False):
        return self.mtcnn.detect(Image.fromarray(bgr[:, :, ::-1]), landmarks=landmarks)


class IdentityBank:
    """신원별 생성 조건(ArcFace 평균 + CLIP 얼굴)과 검증용 CASIA 템플릿 — 정면 사진 최대 N_SRC 장."""

    def __init__(self, m, pool, rng):
        self.m, self.pool, self.rng, self.cache = m, pool, rng, {}

    def get(self, key):
        if key in self.cache:
            return self.cache[key]
        faces = []
        cands = list(self.pool[key])
        self.rng.shuffle(cands)
        for p in cands[:60]:
            im = cv2.imread(p)
            _, _, lm = self.m.detect(im, landmarks=True)
            if lm is None:
                continue
            l = lm[0]
            ed = float(np.hypot(*(l[0] - l[1])))
            mid = (l[0] + l[1]) / 2
            if abs(l[2][0] - mid[0]) < 0.12 * ed and abs(l[0][1] - l[1][1]) < 0.15 * ed and ed > 40:
                faces.append((im, l))
            if len(faces) >= N_SRC:
                break
        if not faces:
            self.cache[key] = None
            return None
        idv, clip_face = self.m.fi.identity_embeds(faces)
        t = self.m.casia_embed([cv2.resize(f[0], (160, 160)) for f in faces]).mean(0)
        self.cache[key] = (idv, clip_face, t / np.linalg.norm(t))
        return self.cache[key]


def insert_person(m, canvas, slot, ident, prompt, ref, rng, stats):
    """canvas 제자리 수정. 성공 시 dict(box, id_cos, sc, ip), 실패 시 None (canvas 원상태)."""
    idv, clip_face, casia_t = ident
    fs, cx, cy = slot_geom(slot)
    cy += 0.15 * fs
    n = int(WIN_MULT * fs)
    H, W = canvas.shape[:2]
    x0, y0 = int(round(cx - n / 2)), int(round(cy - n / 2))
    orig, dst, src = c33.get_window(canvas, x0, y0, n)
    mask = head_mask(SIDE, fs * SIDE / n)
    gmask = cv2.GaussianBlur(cv2.dilate(mask, np.ones((19, 19), np.uint8)), (0, 0), 6)
    win = cv2.resize(orig, (SIDE, SIDE), interpolation=cv2.INTER_LANCZOS4)
    sc, ip = rng.uniform(*SHORTCUT), rng.uniform(*IP_SCALE)
    out = m.fi.inpaint(win, gmask, idv, clip_face, prompt, NEGATIVE, seed=rng.randint(0, 2**31 - 1),
                       steps=STEPS, shortcut_scale=sc, ip_scale=ip)
    gen = cv2.resize(out, (n, n), interpolation=cv2.INTER_AREA)
    mn = cv2.resize(mask, (n, n), interpolation=cv2.INTER_AREA)
    alpha = cv2.GaussianBlur(mn, (0, 0), max(1.0, 0.12 * fs)) * c33.border_taper(n)
    comp = pyramid_blend(harmonize(gen, orig, alpha, fs, ref[0]), orig, alpha)
    # 그린 얼굴 재검출 → 라벨 박스
    big = cv2.resize(comp, (SIDE // 2, SIDE // 2), interpolation=cv2.INTER_CUBIC)
    boxes, probs = m.detect(big)
    if boxes is None:
        stats["no_face_after"] += 1
        return None
    s_ = n / (SIDE // 2)
    c = SIDE // 4
    j = int(np.argmin([abs((b[0] + b[2]) / 2 - c) + abs((b[1] + b[3]) / 2 - (c - 0.15 * fs / s_)) for b in boxes]))
    if probs[j] < 0.9:
        stats["no_face_after"] += 1
        return None
    b = boxes[j] * s_
    box = [float(b[0] + x0), float(b[1] + y0), float(b[2] + x0), float(b[3] + y0)]
    if box[0] < 0 or box[1] < 0 or box[2] > W or box[3] > H:
        stats["face_off_frame"] += 1
        return None
    before = canvas[dst].copy()
    canvas[dst] = comp[src]
    cos = float(m.casia_embed([c33.crop_box(canvas, box)])[0] @ casia_t)
    stats["cos_all"].append(cos)
    if cos < ID_COS_MIN:
        canvas[dst] = before
        stats["id_gate_reject"] += 1
        return None
    return {"box": [round(v, 1) for v in box], "id_cos": round(cos, 3), "sc": round(sc, 2), "ip": round(ip, 2)}


def unknown_genders(pools) -> dict:
    """모르는 사람 성별 (사진 3장 CLIP 다수결) — scripts/41 과 같은 방식, 결과는 캐시."""
    if UGENDER_CACHE.exists():
        return json.loads(UGENDER_CACHE.read_text(encoding="utf-8"))
    clip = genmodels.ClipScorer("cuda")
    labels = ["a photo of a man's face", "a photo of a woman's face"]
    out = {}
    for sp, pool in pools.items():
        for g, ps in pool.items():
            v = [clip.classify(Image.open(p).convert("RGB"), labels)[labels[0]] for p in ps[:3]]
            out[f"{sp}:{g}"] = "M" if np.mean(v) >= 0.5 else "F"
    del clip
    torch.cuda.empty_cache()
    UGENDER_CACHE.write_text(json.dumps(out), encoding="utf-8")
    return out


def identity_glasses(pools) -> dict:
    """신원별 안경 여부 (등록 사진 최대 5장 CLIP 평균). FaceID 신원 벡터(ArcFace)는 안경을 무시해서
    안경을 늘 쓰는 사람(타깃 4번)은 안경 없는 얼굴로 그려지고 신원 검증(CASIA)에서 매번 떨어졌다
    (파일럿 0/수십 회) → 프롬프트에 안경을 명시한다."""
    if GLASSES_CACHE.exists():
        return json.loads(GLASSES_CACHE.read_text(encoding="utf-8"))
    clip = genmodels.ClipScorer("cuda")
    labels = ["a photo of a person wearing eyeglasses", "a photo of a person without glasses"]
    out = {}
    for key, pool in pools.items():
        for who, ps in pool.items():
            v = [clip.classify(Image.open(p).convert("RGB"), labels)[labels[0]] for p in ps[:5]]
            out[f"{key}:{who}"] = bool(np.mean(v) >= 0.6)
    del clip
    torch.cuda.empty_cache()
    GLASSES_CACHE.write_text(json.dumps(out), encoding="utf-8")
    return out


def finish_cctv(canvas, faces, untouched, night, rng, cam):
    """렌즈(배럴+감광) → OSD → CCTV 열화. 모두 프레임 전체에 균일하게. 라벨 박스는 렌즈 변환으로 옮긴다."""
    img, fwd = cs.lens(canvas, rng.uniform(*K1), rng.uniform(*VIGNETTE))
    img = cs.osd(img, rng, cam, night)

    def mv(b):
        xs, ys = zip(*(fwd(x, y) for x, y in ((b[0], b[1]), (b[2], b[1]), (b[0], b[3]), (b[2], b[3]))))
        return [round(float(min(xs)), 1), round(float(min(ys)), 1), round(float(max(xs)), 1), round(float(max(ys)), 1)]

    faces = [{**f, "box_clean": f["box"], "box": mv(f["box"])} for f in faces]
    untouched = [mv(b) for b in untouched]
    img, p = c33.frame_degrade(img, rng)
    return img, faces, untouched, p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-scenes", type=int, default=0, help="split 별 장면 수 제한 (파일럿)")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--out", default="", help="F10 아래 하위 폴더 (파일럿용). 비우면 F10 자체")
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--rerender", type=float, default=0.12,
                    help="삽입 후 프레임 전체 재생성 강도 (0=끔). 얼굴만 확대 창에서 그린 질감 차이를 없앤다")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    OUT = F10 / args.out if args.out else F10

    split = json.loads((F10 / "split.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader(open(F10 / "manifest.csv", encoding="utf-8")))
    tgender = {int(r["cls"]): r["gender"] for r in csv.DictReader(open(F10 / "targets.csv", encoding="utf-8"))}
    v1 = {r["path"]: r for r in csv.DictReader(open(V1 / "manifest.csv", encoding="utf-8"))}
    tgt = {s: defaultdict(list) for s in ("train", "test")}
    for r in rows:
        if r["split"] in ("train", "test_hq"):
            tgt["train" if r["split"] == "train" else "test"][int(r["cls"])].append(str(F10 / r["path"]))
    unk = {s: defaultdict(list) for s in ("train", "test")}
    for sp, key in (("train", "unknown_train"), ("test", "unknown_test_hq")):
        for p in split[key]:
            unk[sp][v1[p]["gid"]].append(str(V1 / p))
    ug = unknown_genders(unk)
    glasses = identity_glasses({f"t{sp}": tgt[sp] for sp in tgt} | {f"u{sp}": unk[sp] for sp in unk})
    print(f"glasses: targets {[c for c in range(10) if glasses.get(f'ttrain:{c}')]}  "
          f"unknown {sum(v for k, v in glasses.items() if k.startswith('u'))}", flush=True)
    fails, wins = defaultdict(int), defaultdict(int)

    m = Models()
    banks = {(k, s): IdentityBank(m, pool[s], rng) for k, pool in (("t", tgt), ("u", unk)) for s in pool}
    scenes = [json.loads(l) for l in open(s50.MANIFEST, encoding="utf-8")]
    stats = defaultdict(int)
    stats["cos_all"] = []
    crop_rows = []
    t0 = time.time()
    for sp in args.splits:
        sc_list = [s for s in scenes if s["split"] == sp]
        if args.max_scenes:
            sc_list = sc_list[:args.max_scenes]
        d = OUT / "cctv_frames" / sp
        (d / "images").mkdir(parents=True, exist_ok=True)
        (d / "labels").mkdir(parents=True, exist_ok=True)
        fj = open(d / "frames.jsonl", "w", encoding="utf-8")
        use_t = defaultdict(int)
        unk_by_g = defaultdict(list)
        for g in unk[sp]:
            unk_by_g[ug[f"{sp}:{g}"]].append(g)
        for si, scene in enumerate(sc_list):
            canvas = cv2.imread(str(s50.OUT_DIR / scene["file"]))
            slots = scene["slots"]
            ref = scene_face_ref(canvas, slots)
            render = [s for s in slots if s["gender"] in ("M", "F")
                      and s50.FACE_PER_EYE * s["eye_dist"] >= MIN_RENDER_FACE][:MAX_RENDER]
            # 타깃 자리: 다시 그릴 슬롯 중 무작위 위치(크기 순서에 묶이지 않게), 성별 맞춤, 덜 쓴 사람 우선
            order = render[:]
            rng.shuffle(order)
            plan, used_t, used_u = [], set(), set()
            for s in order:
                kind, who = "u", None
                if sum(p[0] == "t" for p in plan) < MAX_TARGETS:
                    cands = [c for c in range(10) if tgender[c] == s["gender"] and c not in used_t
                             and not (fails[("t", sp, c)] >= MAX_FAILS and wins[("t", sp, c)] == 0)]
                    if cands:
                        kind, who = "t", min(cands, key=lambda c: (use_t[c], rng.random()))
                        used_t.add(who)
                if kind == "u":
                    who = rng.choice([g for g in unk_by_g[s["gender"]] if g not in used_u])
                    used_u.add(who)
                plan.append((kind, who, s))
            faces, untouched = [], [slot_box(s) for s in slots if all(s is not p[2] for p in plan)]
            for kind, who, s in plan:
                ident = banks[(kind, sp)].get(who)
                g = GENDER_WORD[s["gender"]]
                if glasses.get(f"{kind}{sp}:{who}"):
                    g += " wearing eyeglasses"
                prompt = PROMPT[scene["night"]].format(g=g)
                res = None if ident is None else insert_person(m, canvas, s, ident, prompt, ref, rng, stats)
                (wins if res is not None else fails)[(kind, sp, who)] += 1
                if res is None:
                    untouched.append(slot_box(s))
                    continue
                if kind == "t":
                    use_t[who] += 1
                faces.append({"cls": who if kind == "t" else UNKNOWN, "who": str(who), "gender": s["gender"],
                              "_t": ident[2], **res})
            if not any(f["cls"] != UNKNOWN for f in faces):
                stats["frame_no_target"] += 1
                continue
            if args.rerender > 0:
                canvas = m.fi.rerender(canvas, scene["prompt"], s50.NEGATIVE, seed=rng.randint(0, 2**31 - 1),
                                       strength=args.rerender)
            for f in faces:   # 재생성 후 신원 재검증 — 미달 얼굴은 라벨은 두고 학습 크롭에서만 뺀다 (41b 와 같은 규칙)
                f["id_cos_final"] = round(float(m.casia_embed([c33.crop_box(canvas, f["box"])])[0] @ f.pop("_t")), 3)
                stats["final_ok" if f["id_cos_final"] >= ID_COS_MIN else "final_drop"] += 1
            name = Path(scene["file"]).stem
            img, faces, untouched, p = finish_cctv(canvas, faces, untouched, scene["night"], rng, si % 24 + 1)
            H, W = img.shape[:2]
            cv2.imwrite(str(d / "images" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            with open(d / "labels" / f"{name}.txt", "w", encoding="utf-8") as lf:
                for f in faces:
                    b = f["box"]
                    lf.write(f"{f['cls']} {(b[0] + b[2]) / 2 / W:.6f} {(b[1] + b[3]) / 2 / H:.6f} "
                             f"{(b[2] - b[0]) / W:.6f} {(b[3] - b[1]) / H:.6f}\n")
            fj.write(json.dumps({"image": f"images/{name}.jpg", "scene": scene["file"], "size": [W, H],
                                 "night": scene["night"], "faces": faces, "untouched_frontal": untouched,
                                 "degrade": p.to_dict(), "lens_style": "barrel+vignette+osd", "pipeline": "v4"},
                                ensure_ascii=False) + "\n")
            fj.flush()
            stats[f"{sp}_frames"] += 1
            stats[f"{sp}_targets"] += sum(f["cls"] != UNKNOWN for f in faces)
            stats[f"{sp}_unknowns"] += sum(f["cls"] == UNKNOWN for f in faces)
            if sp == "train":
                for k, f in enumerate(faces):
                    if f["id_cos_final"] < ID_COS_MIN:
                        continue
                    rel = Path("synth_cctv") / f"{f['cls']:02d}" / f"{name}_{k}.jpg"
                    (OUT / rel).parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(OUT / rel), c33.crop_box(img, f["box"]), [cv2.IMWRITE_JPEG_QUALITY, 95])
                    crop_rows.append({"path": rel.as_posix(), "cls": f["cls"], "who": f["who"],
                                      "frame": f"cctv_frames/train/images/{name}.jpg",
                                      "face_px": round(f["box"][2] - f["box"][0], 1), "id_cos": f["id_cos_final"]})
            if si % 10 == 0:
                c = stats["cos_all"]
                summ = {k: v for k, v in stats.items() if not isinstance(v, list)}
                print(f"  [{sp}] scene {si}/{len(sc_list)}  {summ}  id_cos med "
                      f"{np.median(c) if c else float('nan'):.3f}  {time.time() - t0:.0f}s", flush=True)
        fj.close()
        print(f"  [{sp}] target count per class: {dict(sorted(use_t.items()))}", flush=True)
    if crop_rows:
        with open(OUT / "synth_manifest.csv", "w", newline="", encoding="utf-8") as fo:
            wr = csv.DictWriter(fo, fieldnames=list(crop_rows[0]))
            wr.writeheader()
            wr.writerows(crop_rows)
    c = np.array(stats["cos_all"])
    print(f"[done] {({k: v for k, v in stats.items() if not isinstance(v, list)})}  {time.time() - t0:.0f}s")
    if len(c):
        print(f"  id_cos q10 {np.quantile(c, .1):.3f}  med {np.median(c):.3f}  통과율 {(c >= ID_COS_MIN).mean():.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
