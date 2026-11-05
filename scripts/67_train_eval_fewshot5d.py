"""9차 학습·평가 — fewshot5d(사진 5장 × 타깃 10명, 모스크 CCTV 1 + WIDER 4, 나이순 스왑).

[검출] YOLO26n, 클래스 1개(face), COCO 사전학습(yolo26n.pt)에서 시작, imgsz 1280, 기본 증강(Ultralytics 기본값)
  orig      스왑 없는 원본 5장만 (21번 복제해 스텝 수를 swap 과 맞춤)
  swap      원본 5장 + 스왑 변형 100장
  swap_p2   swap 과 같은 데이터, yolo26n-p2(stride 4 층 추가 — 20px 안팎 얼굴용)
  평가: test 를 출처별로 — 모스크(SCRFD 의사 라벨) / WIDER_val(사람 라벨)

[인식] facenet-pytorch InceptionResnetV1 (CASIA-WebFace), CosFace 11클래스(타깃 10 + 모르는 사람), 43 과 같은 설정
  baseline   파인튜닝 없음
  real_deg   타깃 실사진(swap_train) + 모르는 사람 실사진, Albumentations 열화 p=0.5 — 스왑 없이 할 수 있는 최선
  swap       fewshot5d train 의 군중 속 얼굴 크롭(스왑한 타깃·모르는 사람 + 안 바꾼 원래 얼굴 = 모르는 사람)
  추론: 등록 사진 5장 평균 템플릿과 코사인 → 가장 비슷한 타깃 + 임계값.
  임계값: val 프레임의 모르는 얼굴(스왑한 모르는 사람 + 원래 얼굴)을 10명 중 누군가로 착각하는 비율 1%.
  평가: test 프레임 속 타깃 얼굴(박스는 라벨 = 검출 성공 가정)을 맞는 사람으로 찾았는가, 출처·크기별.

출력  runs/fewshot5d/{det_<arm>, rec_<arm>.pt}, reports/eval_fewshot5d.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rareobj import paths  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

t34 = importlib.import_module("34_train_eval_face")
t43 = importlib.import_module("43_train_eval_face10")
D = paths.DATASET / "CCTV_work" / "fewshot5d"
RUNS = paths.ROOT / "runs" / "fewshot5d"
OUT_JSON = paths.REPORTS / "eval_fewshot5d.json"
N_T, UNKNOWN = 10, 10
DET_ARMS = ["orig", "swap", "swap_p2"]
REC_ARMS = ["baseline", "real_deg", "swap"]
EPOCHS, IMGSZ, BATCH = 60, 1280, 8
BATCH_P2 = 4                    # p2 는 1280 에서 batch 8 이 8GB 를 넘겨 공유 메모리로 넘어가 에폭당 9분이 걸렸다
CROP_MARGIN = 14 / 146          # facenet MTCNN margin=14 @160 과 같은 여유 (30 번 데이터와 같은 틀)
FAR = 0.01
SIZE_BINS = [(0, 20), (20, 30), (30, 10_000)]


# ---------------------------------------------------------------- 검출
def det_lists(meta):
    """ultralytics 이미지 목록 파일 — 학습 arm 별 train, 출처별 test."""
    lst = RUNS / "lists"
    lst.mkdir(parents=True, exist_ok=True)
    rep = D / "orig_rep"
    for sub in ("images", "labels"):
        (rep / sub).mkdir(parents=True, exist_ok=True)
    origs = [m for m in meta if m["split"] == "train" and m["image"].endswith("_orig.jpg")]
    rep_imgs = []
    n_rep = sum(m["split"] == "train" for m in meta) // len(origs)
    for m in origs:
        src = D / m["image"]
        for r in range(n_rep):
            dst = rep / "images" / f"{src.stem}_r{r:02d}.jpg"
            if not dst.exists():
                shutil.copy(src, dst)
                shutil.copy(D / m["image"].replace("images/", "labels/").replace(".jpg", ".txt"),
                            rep / "labels" / f"{src.stem}_r{r:02d}.txt")
            rep_imgs.append(dst)
    files = {"train_orig": rep_imgs,
             "train_swap": [D / m["image"] for m in meta if m["split"] == "train"],
             "val": [D / m["image"] for m in meta if m["split"] == "val"]}
    for src in ("cctv", "wider"):
        files[f"test_{src}"] = [D / m["image"] for m in meta if m["split"] == "test" and m["src"] == src]
    out = {}
    for k, fs in files.items():
        p = lst / f"{k}.txt"
        p.write_text("\n".join(Path(f).as_posix() for f in fs), encoding="utf-8")
        out[k] = p
    return out


def det_yaml(train, val, test=None) -> Path:
    p = RUNS / "lists" / f"{Path(train).stem}__{Path(test or val).stem}.yaml"
    p.write_text(f"train: {Path(train).as_posix()}\nval: {Path(val).as_posix()}\n"
                 f"test: {Path(test or val).as_posix()}\nnames:\n  0: face\n", encoding="utf-8")
    return p


def run_det(arm, L) -> dict:
    from ultralytics import YOLO
    w = RUNS / f"det_{arm}" / "weights" / "best.pt"
    if not w.exists():
        model = YOLO("yolo26n-p2.yaml").load("yolo26n.pt") if arm == "swap_p2" else YOLO("yolo26n.pt")
        y = det_yaml(L["train_orig" if arm == "orig" else "train_swap"], L["val"])
        t0 = time.time()
        model.train(data=str(y), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH_P2 if arm == "swap_p2" else BATCH, workers=4, seed=0, deterministic=True,
                    project=str(RUNS), name=f"det_{arm}", exist_ok=True, plots=False, verbose=False,
                    single_cls=True, patience=20)
        print(f"  [det {arm}] 학습 {time.time() - t0:.0f}s", flush=True)
    model = YOLO(str(w))
    res = {}
    for src in ("cctv", "wider"):
        y = det_yaml(L["train_swap"], L["val"], L[f"test_{src}"])
        m = model.val(data=str(y), split="test", imgsz=IMGSZ, batch=BATCH, plots=False, verbose=False,
                      project=str(RUNS), name=f"val_{arm}_{src}", exist_ok=True)
        res[src] = {"mAP50": float(m.box.map50), "mAP50-95": float(m.box.map), "P": float(m.box.mp),
                    "R": float(m.box.mr)}
    print(f"  [det {arm}] {res}", flush=True)
    return res


# ---------------------------------------------------------------- 인식
def crop160(img, box):
    x0, y0, x1, y1 = box
    mx, my = (x1 - x0) * CROP_MARGIN / 2, (y1 - y0) * CROP_MARGIN / 2
    H, W = img.shape[:2]
    a, b = int(max(x0 - mx, 0)), int(max(y0 - my, 0))
    c, d = int(min(x1 + mx, W)), int(min(y1 + my, H))
    return cv2.resize(img[b:d, a:c], (160, 160), interpolation=cv2.INTER_CUBIC)


def build_crops(meta):
    """프레임 속 모든 라벨 얼굴 크롭. 스왑한 얼굴은 cls(타깃 0~9 / 모르는 사람 10), 나머지 원래 얼굴은 10."""
    idx = D / "crops" / "index.json"
    if idx.exists():
        return json.loads(idx.read_text(encoding="utf-8"))
    out = []
    for m in meta:
        if m["image"].endswith("_orig.jpg"):
            continue
        img = cv2.imread(str(D / m["image"]))
        H, W = img.shape[:2]
        sw = {tuple(np.round(f["box"], 1)): f for f in m["swaps"]}
        lines = (D / m["image"].replace("images/", "labels/").replace(".jpg", ".txt")).read_text().split("\n")
        d = D / "crops" / m["split"]
        d.mkdir(parents=True, exist_ok=True)
        boxes = []
        for l in filter(None, lines):
            _, cx, cy, w, h = map(float, l.split())
            boxes.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
        # 스왑 얼굴은 meta 의 박스를 쓰고, 라벨 박스 중 스왑 얼굴과 겹치는 것은 뺀다
        faces = [(f["box"], f["cls"], True) for f in m["swaps"]]
        for b in boxes:
            if all(_iou(b, f[0]) < 0.3 for f in faces):
                faces.append((b, UNKNOWN, False))
        for k, (b, c, swapped) in enumerate(faces):
            if b[2] - b[0] < 8 or b[3] - b[1] < 8:
                continue
            p = d / f"{Path(m['image']).stem}_{k:03d}.jpg"
            cv2.imwrite(str(p), crop160(img, b), [cv2.IMWRITE_JPEG_QUALITY, 95])
            out.append({"path": str(p), "cls": int(c), "swapped": swapped, "split": m["split"], "src": m["src"],
                        "w": float(b[2] - b[0])})
    idx.write_text(json.dumps(out), encoding="utf-8")
    return out


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def rec_items(crops, T, ids):
    tr = [c for c in crops if c["split"] == "train"]
    swap_t = [(c["path"], c["cls"]) for c in tr if c["cls"] < UNKNOWN]
    swap_u = [(c["path"], UNKNOWN) for c in tr if c["cls"] == UNKNOWN]
    real_t = [(p, t["cls"]) for t in T for p in t["swap_train"]]
    real_u = [(p, UNKNOWN) for u in ids["unknown_train"] for p in u["imgs"]]
    return swap_t, swap_u, real_t, real_u


def train_rec(arm, swap_t, swap_u, real_t, real_u):
    # 43.train 의 학습 루프를 그대로 쓰고 arm 이름만 맞춘다: real_deg = 실사진, synth_cctv = 스왑 크롭
    t43.RUNS = RUNS
    if arm == "real_deg":
        return t43.train("real_deg", 0, real_t, real_u, [], [])
    return t43.train("synth_cctv", 0, [], [], swap_t, swap_u)


def eval_rec(model, crops, T):
    E_en = t34.embed(model, [p for t in T for p in t["enroll"]])
    Tm = np.stack([np.mean([E_en[p] for p in t["enroll"]], 0) for t in T])
    Tm /= np.linalg.norm(Tm, axis=1, keepdims=True)
    ev = [c for c in crops if c["split"] in ("val", "test")]
    E = t34.embed(model, [c["path"] for c in ev])
    S = np.stack([E[c["path"]] for c in ev]) @ Tm.T
    smax, sarg = S.max(1), S.argmax(1)
    cls = np.array([c["cls"] for c in ev])
    sp = np.array([c["split"] for c in ev])
    src = np.array([c["src"] for c in ev])
    w = np.array([c["w"] for c in ev])
    res = {}
    for s in ("cctv", "wider"):
        thr = float(np.quantile(smax[(sp == "val") & (src == s) & (cls == UNKNOWN)], 1 - FAR))
        te = (sp == "test") & (src == s)
        tg, un = te & (cls < UNKNOWN), te & (cls == UNKNOWN)
        hit = (sarg == cls) & (smax >= thr)
        r = {"threshold@far1%(val)": thr, "unknown_far(test)": float((smax[un] >= thr).mean()),
             "top1": float((sarg[tg] == cls[tg]).mean()), "find@far1%": float(hit[tg].mean()),
             "n_target": int(tg.sum()), "n_unknown": int(un.sum())}
        for lo, hi in SIZE_BINS:
            b = tg & (w >= lo) & (w < hi)
            if b.sum():
                r[f"find@far1%_{lo}-{hi if hi < 10_000 else ''}px"] = [round(float(hit[b].mean()), 3), int(b.sum())]
        res[s] = r
    return res


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", nargs="*", default=DET_ARMS)
    ap.add_argument("--rec", nargs="*", default=REC_ARMS)
    args = ap.parse_args()
    meta = [json.loads(l) for l in open(D / "meta.jsonl", encoding="utf-8")]
    T = json.loads((D / "targets.json").read_text(encoding="utf-8"))
    ids = json.loads((paths.DATASET / "CCTV_work" / "fewshot5" / "identities.json").read_text(encoding="utf-8"))
    for t, t_full in zip(T, ids["targets"]):
        t["swap_train"] = t_full["swap_train"]          # targets.json 은 소스 첫 장만 저장 — 전체 목록은 캐시에서
    results = json.loads(OUT_JSON.read_text(encoding="utf-8")) if OUT_JSON.exists() else {}
    RUNS.mkdir(parents=True, exist_ok=True)

    crops = build_crops(meta)
    swap_t, swap_u, real_t, real_u = rec_items(crops, T, ids)
    print(f"크롭 {len(crops)}  학습: 스왑 타깃 {len(swap_t)} · 군중 모르는 얼굴 {len(swap_u)}"
          f" / 실사진 타깃 {len(real_t)} · 모르는 사람 {len(real_u)}", flush=True)
    for arm in args.rec:
        key = f"rec/{arm}"
        if key in results:
            print(f"[skip] {key}")
            continue
        t0 = time.time()
        model = t34.load_backbone().cuda().eval() if arm == "baseline" else train_rec(arm, swap_t, swap_u, real_t, real_u)
        results[key] = eval_rec(model.eval(), crops, T)
        results[key]["sec"] = round(time.time() - t0)
        print(f"[rec {arm}] {json.dumps(results[key], ensure_ascii=False)}", flush=True)
        OUT_JSON.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
        del model
        torch.cuda.empty_cache()

    L = det_lists(meta)
    for arm in args.det:
        key = f"det/{arm}"
        if key in results:
            print(f"[skip] {key}")
            continue
        t0 = time.time()
        results[key] = run_det(arm, L)
        results[key]["sec"] = round(time.time() - t0)
        OUT_JSON.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
        torch.cuda.empty_cache()
    print("[done]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
