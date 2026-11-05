"""9차 — 1단계: fewshot5e(변형 10배) + 640 조각 학습 + YOLO26s → det_mc3.

det_mc2(YOLO26n, 1280 전체 장면, 변형 105장)는 학습 이미지에서도 타깃 클래스를 60% 만 맞혔다.
  데이터   fewshot5e — 같은 사진 5장, 변형 200장씩 (scripts/70)
  조각     추론 때 1280 으로 맞춰지는 크기로 장면을 키운 뒤 640×640 조각을 자른다(얼굴 화소 크기는 추론 때와 같다).
           장마다 타깃 얼굴을 중심(±흔들기)으로 1개 + 무작위 1개. 메모리가 1/4 이라 YOLO26s 를 돌릴 수 있다.
  열화     조각 절반에 약한 열화(모션블러 3~5, 축소 0.6~0.9, JPEG 40~90) — 20px 얼굴을 지우지 않는 범위
  평가     68 과 같은 함수 — 전체 장면 1280 추론, fewshot5d 와 같은 val/test, SDXL 시험셋

출력  data/dataset/CCTV_work/fewshot5e_mc/, fewshot5e_crops/, runs/fewshot5d/det_mc3/,
      reports/eval_fewshot5d.json["det_mc3"]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from importlib import import_module
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
m68 = import_module("68_yolo_identity")
m68.SRC, m68.MC = m68.WORK / "fewshot5e", m68.WORK / "fewshot5e_mc"
CROPS = m68.WORK / "fewshot5e_crops"
ARM, MODEL = "det_mc3", "yolo26s.pt"
CROP, EPOCHS, BATCH = 640, 80, 16
HP = dict(cls_pw=0.5, cls=1.5, scale=0.2, hsv_s=0.3, patience=0)
DEG_P = 0.5


def degrader():
    import albumentations as A
    return A.Compose([A.MotionBlur(blur_limit=(3, 5), p=0.7),
                      A.Downscale(scale_min=0.6, scale_max=0.9, interpolation=1, p=0.7),
                      A.ImageCompression(quality_lower=40, quality_upper=90, p=0.7)])


def read_labels(p, W, H):
    out = []
    for l in filter(None, p.read_text().split("\n")):
        c, cx, cy, w, h = l.split()
        cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
        out.append((int(c), cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return out


def write_crop(name, big, L, x0, y0, sp, deg, rng):
    H, W = big.shape[:2]
    x0, y0 = int(np.clip(x0, 0, max(0, W - CROP))), int(np.clip(y0, 0, max(0, H - CROP)))
    im = np.full((CROP, CROP, 3), 114, np.uint8)
    a = big[y0:y0 + CROP, x0:x0 + CROP]
    im[:a.shape[0], :a.shape[1]] = a
    if deg is not None and rng.random() < DEG_P:
        im = deg(image=im)["image"]
    lines = []
    for c, bx0, by0, bx1, by1 in L:
        cx0, cy0 = max(bx0 - x0, 0), max(by0 - y0, 0)
        cx1, cy1 = min(bx1 - x0, a.shape[1]), min(by1 - y0, a.shape[0])
        if cx1 - cx0 < 3 or cy1 - cy0 < 3:
            continue
        if (cx1 - cx0) * (cy1 - cy0) < 0.4 * (bx1 - bx0) * (by1 - by0):
            continue
        lines.append(f"{c} {(cx0 + cx1) / 2 / CROP:.6f} {(cy0 + cy1) / 2 / CROP:.6f} "
                     f"{(cx1 - cx0) / CROP:.6f} {(cy1 - cy0) / CROP:.6f}")
    cv2.imwrite(str(CROPS / "images" / sp / f"{name}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    (CROPS / "labels" / sp / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")


def build_crops(metas):
    rng = np.random.default_rng(0)
    deg = degrader()
    for sp in ("train", "val"):
        (CROPS / "images" / sp).mkdir(parents=True, exist_ok=True)
        (CROPS / "labels" / sp).mkdir(parents=True, exist_ok=True)
    n = {"train": 0, "val": 0}
    for m in metas:
        if m["split"] not in n:
            continue
        img = cv2.imread(str(m68.MC / m["image"]))
        H, W = img.shape[:2]
        s = m68.IMGSZ / max(H, W)
        big = cv2.resize(img, (round(W * s), round(H * s)), interpolation=cv2.INTER_LINEAR)
        L = [(c, x0 * s, y0 * s, x1 * s, y1 * s) for c, x0, y0, x1, y1 in
             read_labels(m68.MC / m["image"].replace("images/", "labels/").replace(".jpg", ".txt"), W, H)]
        bh, bw = big.shape[:2]
        stem = Path(m["image"]).stem
        if m["split"] == "train":
            tg = [b for b in L if b[0] < m68.N_T]
            if tg:
                c, x0, y0, x1, y1 = tg[int(rng.integers(len(tg)))]
                j = rng.uniform(-0.35, 0.35, 2) * CROP
                write_crop(f"{stem}_t", big, L, (x0 + x1) / 2 - CROP / 2 + j[0], (y0 + y1) / 2 - CROP / 2 + j[1],
                           "train", deg, rng)
            write_crop(f"{stem}_r", big, L, rng.uniform(0, max(1, bw - CROP)), rng.uniform(0, max(1, bh - CROP)),
                       "train", deg, rng)
            n["train"] += 1 + bool(tg)
        else:                                                   # val: 겹치지 않게 격자
            for yi, y0 in enumerate(range(0, max(1, bh - CROP // 4), CROP)):
                for xi, x0 in enumerate(range(0, max(1, bw - CROP // 4), CROP)):
                    write_crop(f"{stem}_{yi}{xi}", big, L, x0, y0, "val", None, rng)
                    n["val"] += 1
    y = CROPS / "data.yaml"
    y.write_text(f"path: {CROPS.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
                 + "".join(f"  {k}: {v}\n" for k, v in m68.NAMES.items()), encoding="utf-8")
    print("[crops]", n, flush=True)
    return y


def main() -> int:
    global CROPS, ARM, EPOCHS
    from ultralytics import YOLO

    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--data", default="fewshot5e", help="72 의 fewshot5f_v4 / fewshot5f_mix 등")
    ap.add_argument("--arm", default=ARM)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    m68.SRC, m68.MC = m68.WORK / args.data, m68.WORK / f"{args.data}_mc"
    CROPS, ARM, EPOCHS = m68.WORK / f"{args.data}_crops", args.arm, args.epochs
    metas, _ = m68.build_mc()
    w = m68.RUNS / ARM / "weights" / "best.pt"
    if not args.eval_only and not w.exists():
        y = build_crops(metas)
        t0 = time.time()
        YOLO(MODEL).train(data=str(y), imgsz=CROP, batch=BATCH, epochs=EPOCHS, workers=4, seed=0,
                          deterministic=True, project=str(m68.RUNS), name=ARM, exist_ok=True, plots=False,
                          verbose=False, **HP)
        print(f"[{ARM}] 학습 {time.time() - t0:.0f}s", flush=True)
    model = YOLO(str(w))
    tr = m68.predict_scores(model, m68.MC, [m for m in metas if m["split"] == "train"][::5])
    tg = [r for r in tr if r[0] < m68.N_T]
    print(f"[{ARM}] 학습 이미지 타깃 클래스 정답 {np.mean([r[1] == r[0] for r in tg]):.3f} (n{len(tg)})", flush=True)
    val_rows = m68.predict_scores(model, m68.MC, [m for m in metas if m["split"] == "val"])
    res = {"train_class_correct": float(np.mean([r[1] == r[0] for r in tg])),
           "test_swap": m68.find_metrics(val_rows, m68.predict_scores(
               model, m68.MC, [m for m in metas if m["split"] == "test"]))}
    for key, root in (("test_sdxl", m68.SDXL), ("test_sd15", m68.WORK / "fewshot5d_sd15_test")):
        if (root / "meta.jsonl").exists():
            sm = [json.loads(l) for l in open(root / "meta.jsonl", encoding="utf-8")]
            res[key] = m68.find_metrics(val_rows, m68.predict_scores(model, root, sm))
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    allr = json.loads(m68.OUT_JSON.read_text(encoding="utf-8")) if m68.OUT_JSON.exists() else {}
    allr[ARM] = {**allr.get(ARM, {}), **res}
    m68.OUT_JSON.write_text(json.dumps(allr, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
