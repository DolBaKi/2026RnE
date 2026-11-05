"""9차 — YOLO 만으로 특정인 찾기 (조교 피드백 2026-11-03).

"장면 속 얼굴 박스 자리에 생성 모델로 얼굴 A 를 넣어 만든 데이터로 YOLO 를 학습하면, 추론 때 YOLO 하나로
얼굴 A 를 검출한다." → fewshot5d 를 클래스 11개(타깃 0~9, 그 밖의 얼굴 10)로 다시 라벨해 학습한다.
모르는 사람도 같은 생성 모델(inswapper)로 넣어 두었으므로 "생성된 티 = 타깃" 지름길은 막혀 있다.

평가 — 타깃 얼굴 하나하나를 "맞는 클래스로 찾았는가" (find@FAR1%)
  예측 중 IoU ≥ 0.5 로 가장 잘 맞는 박스의 클래스·신뢰도를 그 얼굴의 결과로 본다.
  임계값: val 의 타깃 아닌 얼굴이 타깃 클래스로 τ 이상 잡히는 비율이 1% 인 τ.
  test_swap   fewshot5d test (inswapper — 학습과 같은 생성 모델이라 지름길을 못 걸러낸다)
  test_sdxl   scripts/69 — 다른 생성 모델(RealVisXL + FaceID-SDXL)로 학습에 안 쓴 사진의 타깃을 넣은 장면
              (생성 모델 간 일반화: 사람을 배웠는지, inswapper 의 그림체를 배웠는지 가른다)

출력  data/dataset/CCTV_work/fewshot5d_mc/, runs/fewshot5d/det_mc/, reports/eval_fewshot5d.json["det_mc"]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rareobj import paths  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

WORK = paths.DATASET / "CCTV_work"
SRC, MC = WORK / "fewshot5d", WORK / "fewshot5d_mc"
SDXL = WORK / "fewshot5d_sdxl_test"
RUNS = paths.ROOT / "runs" / "fewshot5d"
OUT_JSON = paths.REPORTS / "eval_fewshot5d.json"
N_T, OTHER = 10, 10
NAMES = {**{i: f"T{i}" for i in range(N_T)}, OTHER: "face"}
EPOCHS, IMGSZ, BATCH = 60, 1280, 8
# det_mc 는 타깃 클래스를 한 번도 예측하지 않았다(학습 이미지에서도 0%) — "그 밖의 얼굴" 15710 vs 타깃당 ~100,
# 160:1 불균형으로 "face" 만 내도 손실이 낮다. det_mc2 = 역빈도^0.5 클래스 가중치 + cls 손실 3배 + 조기 종료 없이 150 에폭
# + 신원을 지우는 크기·채도 증강 약하게.
ARMS = {"det_mc": {},
        "det_mc2": dict(cls_pw=0.5, cls=1.5, epochs=150, patience=0, scale=0.2, hsv_s=0.3)}
FAR = 0.01


def iou(a, B):
    B = np.asarray(B, np.float32).reshape(-1, 4)
    x0, y0 = np.maximum(a[0], B[:, 0]), np.maximum(a[1], B[:, 1])
    x1, y1 = np.minimum(a[2], B[:, 2]), np.minimum(a[3], B[:, 3])
    i = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - i + 1e-9)


def gt_faces(root, m, W, H):
    """라벨 박스(xyxy) + 클래스: 타깃 스왑 얼굴은 0~9, 나머지(모르는 사람 스왑·원래 얼굴)는 10."""
    lines = (root / m["image"].replace("images/", "labels/").replace(".jpg", ".txt")).read_text().split("\n")
    boxes = []
    for l in filter(None, lines):
        _, cx, cy, w, h = map(float, l.split()[:5])
        boxes.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
    cls = [OTHER] * len(boxes)
    for f in m["swaps"]:
        if f["cls"] < N_T and boxes:
            o = iou(np.float32(f["box"]), boxes)
            if o.max() >= 0.3:
                cls[int(o.argmax())] = f["cls"]
    return np.float32(boxes).reshape(-1, 4), np.int32(cls)


def build_mc():
    import cv2

    meta = [json.loads(l) for l in open(SRC / "meta.jsonl", encoding="utf-8")]
    out = []
    for m in meta:
        img = SRC / m["image"]
        H, W = cv2.imread(str(img)).shape[:2]
        B, C = gt_faces(SRC, m, W, H)
        dst = MC / m["image"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        lab = MC / m["image"].replace("images/", "labels/").replace(".jpg", ".txt")
        lab.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy(img, dst)
        lab.write_text("\n".join(f"{c} {(b[0] + b[2]) / 2 / W:.6f} {(b[1] + b[3]) / 2 / H:.6f} "
                                 f"{(b[2] - b[0]) / W:.6f} {(b[3] - b[1]) / H:.6f}" for b, c in zip(B, C)),
                       encoding="utf-8")
        out.append({**m, "W": W, "H": H})
    lst = {}
    for sp in ("train", "val", "test"):
        p = MC / f"{sp}.txt"
        p.write_text("\n".join((MC / m["image"]).as_posix() for m in out if m["split"] == sp), encoding="utf-8")
        lst[sp] = p
    y = MC / "data.yaml"
    y.write_text(f"train: {lst['train'].as_posix()}\nval: {lst['val'].as_posix()}\ntest: {lst['test'].as_posix()}\n"
                 "names:\n" + "".join(f"  {k}: {v}\n" for k, v in NAMES.items()), encoding="utf-8")
    return out, y


def predict_scores(model, root, metas):
    """GT 얼굴마다 (참 클래스, 맞춘 박스의 예측 클래스, 신뢰도, 폭). 못 맞추면 예측 클래스 -1."""
    rows = []
    for m in metas:
        r = model.predict(str(root / m["image"]), imgsz=IMGSZ, conf=0.01, verbose=False, max_det=1000)[0]
        pb = r.boxes.xyxy.cpu().numpy()
        pc = r.boxes.cls.cpu().numpy().astype(int)
        pf = r.boxes.conf.cpu().numpy()
        B, C = gt_faces(root, m, r.orig_shape[1], r.orig_shape[0])
        for b, c in zip(B, C):
            if len(pb):
                o = iou(b, pb)
                j = int(o.argmax())
                if o[j] >= 0.5:
                    rows.append((int(c), int(pc[j]), float(pf[j]), float(b[2] - b[0]), m.get("src", "")))
                    continue
            rows.append((int(c), -1, 0.0, float(b[2] - b[0]), m.get("src", "")))
    return rows


def find_metrics(val_rows, test_rows):
    """τ: val 의 타깃 아닌 얼굴이 타깃 클래스로 τ 이상 잡히는 비율 = FAR."""
    v = np.array([r[2] if (r[0] == OTHER and 0 <= r[1] < N_T) else 0.0 for r in val_rows])
    tau = float(np.quantile(v[[r[0] == OTHER for r in val_rows]], 1 - FAR))
    res = {}
    for src in sorted({r[4] for r in test_rows}):
        R = [r for r in test_rows if r[4] == src]
        tg = [r for r in R if r[0] < N_T]
        un = [r for r in R if r[0] == OTHER]
        hit = np.array([r[1] == r[0] and r[2] >= tau and r[2] > 0 for r in tg])
        top = np.array([r[1] == r[0] for r in tg])
        w = np.array([r[3] for r in tg])
        d = {"tau": tau, "find@far1%": float(hit.mean()) if len(tg) else None,
             "class_correct_any_conf": float(top.mean()) if len(tg) else None,
             "detected_as_any_class": float(np.mean([r[1] >= 0 for r in tg])) if len(tg) else None,
             "unknown_far": float(np.mean([0 <= r[1] < N_T and r[2] >= tau for r in un])) if un else None,
             "n_target": len(tg), "n_other": len(un)}
        for lo, hi in ((0, 20), (20, 30), (30, 1e9)):
            b = (w >= lo) & (w < hi)
            if b.sum():
                d[f"find_{lo}-{int(hi) if hi < 1e9 else ''}px"] = [round(float(hit[b].mean()), 3), int(b.sum())]
        res[src] = d
    return res


def main() -> int:
    from ultralytics import YOLO

    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--arm", default="det_mc", choices=list(ARMS))
    ap.add_argument("--resume", action="store_true", help="끊긴 학습을 last.pt 에서 이어서")
    args = ap.parse_args()
    metas, y = build_mc()
    w = RUNS / args.arm / "weights" / "best.pt"
    if args.resume:
        t0 = time.time()
        YOLO(str(w.with_name("last.pt"))).train(resume=True)
        print(f"[{args.arm}] 이어서 학습 {time.time() - t0:.0f}s", flush=True)
    elif not args.eval_only and not w.exists():
        t0 = time.time()
        hp = {"epochs": EPOCHS, "patience": 20, **ARMS[args.arm]}
        YOLO("yolo26n.pt").train(data=str(y), imgsz=IMGSZ, batch=BATCH, workers=4, seed=0, deterministic=True,
                                 project=str(RUNS), name=args.arm, exist_ok=True, plots=False, verbose=False, **hp)
        print(f"[{args.arm}] 학습 {time.time() - t0:.0f}s", flush=True)
    model = YOLO(str(w))
    val_rows = predict_scores(model, MC, [m for m in metas if m["split"] == "val"])
    res = {"test_swap": find_metrics(val_rows, predict_scores(model, MC, [m for m in metas if m["split"] == "test"]))}
    if (SDXL / "meta.jsonl").exists():
        sm = [json.loads(l) for l in open(SDXL / "meta.jsonl", encoding="utf-8")]
        res["test_sdxl"] = find_metrics(val_rows, predict_scores(model, SDXL, sm))
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    allr = json.loads(OUT_JSON.read_text(encoding="utf-8")) if OUT_JSON.exists() else {}
    allr[args.arm] = res
    OUT_JSON.write_text(json.dumps(allr, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
