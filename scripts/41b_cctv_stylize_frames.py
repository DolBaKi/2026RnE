"""8차(10명) — 합성 프레임을 "CCTV 화면"으로 만드는 단계 (생성형 스타일화).

사용자 요청: CCTV 화면으로 보이게 세팅하는 생성형 모델 도입. scripts/41 의 깨끗한 프레임에
  1. 생성형 톤 변환  SDXL-Turbo img2img(strength 0.35)로 감시카메라 녹화 톤을 생성하고 그
     저주파 성분만 원해상도에 옮긴다 (src/rareobj/cctv_style.py — 얼굴 고주파 보존).
     시험에서 strength 0.55 는 초록 얼룩 같은 인공물이 생겨 0.35 로 정했다.
  2. 광학  배럴 왜곡(k1 0.04~0.12) + 주변 감광 — 라벨 박스도 같은 변환으로 옮긴다
  3. OSD  날짜·시각(야간 장면이면 밤 시간), CAM 번호, REC
  4. CCTV 열화  scripts/33 frame_degrade (모션블러·센서 해상도·노이즈·JPEG)
  5. 신원 재검증  스타일화(+렌즈) 전후, 열화 전 같은 얼굴 크롭의 CASIA 코사인(style_cos). train 은
     STYLE_COS_MIN 미만 얼굴을 학습 크롭에서 뺀다(라벨은 기록).

출력  data/dataset/faces10/cctv_frames/{train,test}/images, labels(YOLO), frames.jsonl
      data/dataset/faces10/synth_cctv/<cls>/*.jpg + synth_manifest.csv  (train 얼굴 크롭)
"""

from __future__ import annotations

import csv
import gc
import importlib
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rareobj import cctv_style as cs, genmodels, paths  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

c33 = importlib.import_module("33_compose_face_cctv_synth")
F10 = paths.DATASET / "faces10"
UNKNOWN = 10
STRENGTH = 0.35
K1 = (0.04, 0.12)
VIGNETTE = (0.15, 0.35)
STYLE_COS_MIN = 0.50


def main() -> int:
    from facenet_pytorch import InceptionResnetV1

    rng = random.Random(20261002)
    np.random.seed(20261002)
    te = genmodels.load_sdxl_turbo_text_encoders()
    with torch.no_grad():
        emb = {p: tuple(t.cpu() for t in te.encode_prompt(p, device="cuda", do_classifier_free_guidance=False)[::2])
               for p in cs.STYLE_PROMPTS.values()}
    del te
    gc.collect()
    torch.cuda.empty_cache()
    st = cs.CCTVStylizer(emb)
    casia = InceptionResnetV1(pretrained="casia-webface").eval().cuda()

    @torch.no_grad()
    def cos(a, b):
        x = torch.stack([torch.from_numpy((c[:, :, ::-1].astype(np.float32) - 127.5) / 128).permute(2, 0, 1)
                         for c in (a, b)]).cuda()
        e = F.normalize(casia(x)).cpu().numpy()
        return float(e[0] @ e[1])

    crop_rows, t0 = [], time.time()
    cams = {}
    for sp in ("train", "test"):
        d = F10 / "cctv_frames" / sp
        (d / "images").mkdir(exist_ok=True)
        (d / "labels").mkdir(exist_ok=True)
        frames = [json.loads(l) for l in open(d / "frames_clean.jsonl", encoding="utf-8")]
        fj = open(d / "frames.jsonl", "w", encoding="utf-8")
        n_drop = 0
        for i, fr in enumerate(frames):
            clean = cv2.imread(str(d / fr["image"]))
            img = st.tone(clean, fr["night"], STRENGTH, seed=rng.randint(0, 2**31 - 1))
            img, fwd = cs.lens(img, rng.uniform(*K1), rng.uniform(*VIGNETTE))
            cam = cams.setdefault(fr["scene"], len(cams) % 24 + 1)
            img = cs.osd(img, rng, cam, fr["night"])
            H, W = img.shape[:2]
            faces = []
            for fc in fr["faces"]:
                b = fc["box"]
                xs, ys = zip(*(fwd(x, y) for x, y in ((b[0], b[1]), (b[2], b[1]), (b[0], b[3]), (b[2], b[3]))))
                nb = [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
                # 생성형 스타일화(+렌즈)의 신원 영향만 본다 — 열화 뒤에 재면 의도한 CCTV 열화로
                # 떨어진 어려운 얼굴까지 학습에서 빠진다(시험에서 확인)
                sc_ = cos(c33.crop_box(img, nb), c33.crop_box(clean, b))
                faces.append({**fc, "box_clean": b, "box": [round(v, 1) for v in nb], "style_cos": round(sc_, 3)})
            img, p = c33.frame_degrade(img, rng)
            unt = []
            for b in fr["untouched_frontal"]:
                xs, ys = zip(*(fwd(x, y) for x, y in ((b[0], b[1]), (b[2], b[3]))))
                unt.append([round(float(v), 1) for v in (xs[0], ys[0], xs[1], ys[1])])
            name = Path(fr["image"]).stem
            cv2.imwrite(str(d / "images" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            with open(d / "labels" / f"{name}.txt", "w", encoding="utf-8") as lf:
                for fc in faces:
                    b = fc["box"]
                    lf.write(f"{fc['cls']} {(b[0] + b[2]) / 2 / W:.6f} {(b[1] + b[3]) / 2 / H:.6f} "
                             f"{(b[2] - b[0]) / W:.6f} {(b[3] - b[1]) / H:.6f}\n")
            fj.write(json.dumps({**fr, "image": f"images/{name}.jpg", "faces": faces, "untouched_frontal": unt,
                                 "cam": cam, "degrade": p.to_dict(), "lens_style": "barrel+vignette+osd"},
                                ensure_ascii=False) + "\n")
            if sp == "train":
                for k, fc in enumerate(faces):
                    if fc["style_cos"] < STYLE_COS_MIN:
                        n_drop += 1
                        continue
                    rel = Path("synth_cctv") / f"{fc['cls']:02d}" / f"{name}_{k}.jpg"
                    (F10 / rel).parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(F10 / rel), c33.crop_box(img, fc["box"]), [cv2.IMWRITE_JPEG_QUALITY, 95])
                    crop_rows.append({"path": rel.as_posix(), "cls": fc["cls"], "who": fc["who"],
                                      "frame": f"cctv_frames/train/images/{name}.jpg",
                                      "face_px": round(fc["box"][2] - fc["box"][0], 1), "id_cos": fc["id_cos"]})
            if i % 50 == 0:
                print(f"  [{sp}] {i}/{len(frames)}  {time.time() - t0:.0f}s", flush=True)
        fj.close()
        sc_all = [f["style_cos"] for l in open(d / "frames.jsonl", encoding="utf-8") for f in json.loads(l)["faces"]]
        print(f"[{sp}] frames {len(frames)}  style_cos med {np.median(sc_all):.3f} q10 {np.quantile(sc_all, .1):.3f}"
              f"  train crops dropped {n_drop}")
    with open(F10 / "synth_manifest.csv", "w", newline="", encoding="utf-8") as fo:
        wr = csv.DictWriter(fo, fieldnames=list(crop_rows[0]))
        wr.writeheader()
        wr.writerows(crop_rows)
    print(f"[done] train crops {len(crop_rows)}  {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
