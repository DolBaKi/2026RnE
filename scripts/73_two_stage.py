"""9차 — 2단계 구조 평가 (새 학습 없음): YOLO 로 얼굴 위치 → 인식 모델로 신원.

YOLO 단독(68·71)은 학습에 쓴 생성기에서만 잘 찾는다(inswapper 52% → SD1.5 8~14%). 같은 시험셋에서
  검출   det_mc3 의 박스만 쓴다(클래스 무시, conf ≥ DET_CONF)
  인식   67 의 real_deg (실사진 + Albumentations 열화로 학습, 생성 데이터 안 봄) — 등록 사진 평균 템플릿과 코사인
  기준   68 과 같다 — 타깃 얼굴마다 IoU ≥ 0.5 로 맞는 박스의 판정, 임계값은 val 의 타깃 아닌 얼굴 FAR 1% (출처별)
출력  reports/eval_fewshot5d.json["two_stage"]
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
m68 = importlib.import_module("68_yolo_identity")
m67 = importlib.import_module("67_train_eval_fewshot5d")
t34 = m67.t34
DET_CONF = 0.1
SETS = {"test_swap": m68.WORK / "fewshot5d", "test_sdxl": m68.WORK / "fewshot5d_sdxl_test",
        "test_sd15": m68.WORK / "fewshot5d_sd15_test"}


@torch.no_grad()
def embed(model, crops):
    out = []
    for i in range(0, len(crops), 256):
        x = torch.stack([t34.to_tensor(c) for c in crops[i:i + 256]]).cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(F.normalize(model(x).float()).cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 512), np.float32)


def rows_of(det, rec, Tm, root, metas):
    """GT 얼굴마다 (참 클래스, 예측 신원, 최대 cos, 폭, 출처). 박스를 못 맞추면 예측 -1."""
    rows, crops, idx = [], [], []
    for m in metas:
        img = cv2.imread(str(root / m["image"]))
        r = det.predict(img, imgsz=m68.IMGSZ, conf=DET_CONF, verbose=False, max_det=1000)[0]
        pb = r.boxes.xyxy.cpu().numpy()
        B, C = m68.gt_faces(root, m, img.shape[1], img.shape[0])
        for b, c in zip(B, C):
            j = -1
            if len(pb):
                o = m68.iou(b, pb)
                j = int(o.argmax()) if o.max() >= 0.5 else -1
            rows.append([int(c), -1, 0.0, float(b[2] - b[0]), m["src"]])
            if j >= 0:
                crops.append(m67.crop160(img, pb[j]))
                idx.append(len(rows) - 1)
    S = embed(rec, crops) @ Tm.T
    for k, i in enumerate(idx):
        rows[i][1], rows[i][2] = int(S[k].argmax()), float(S[k].max())
    return rows


def metrics(val, test):
    res = {}
    for src in sorted({r[4] for r in test}):
        v = np.array([r[2] for r in val if r[4] == src and r[0] == m68.OTHER])
        tau = float(np.quantile(v, 1 - m68.FAR))
        R = [r for r in test if r[4] == src]
        tg = [r for r in R if r[0] < m68.N_T]
        un = [r for r in R if r[0] == m68.OTHER]
        hit = np.array([r[1] == r[0] and r[2] >= tau for r in tg])
        w = np.array([r[3] for r in tg])
        d = {"tau": tau, "find@far1%": float(hit.mean()), "top1": float(np.mean([r[1] == r[0] for r in tg])),
             "unknown_far": float(np.mean([r[2] >= tau for r in un])), "n_target": len(tg)}
        for lo, hi in ((0, 20), (20, 30), (30, 1e9)):
            b = (w >= lo) & (w < hi)
            if b.sum():
                d[f"find_{lo}-{int(hi) if hi < 1e9 else ''}px"] = [round(float(hit[b].mean()), 3), int(b.sum())]
        res[src] = d
    return res


def main() -> int:
    from ultralytics import YOLO

    det = YOLO(str(m68.RUNS / "det_mc3" / "weights" / "best.pt"))
    rec = t34.load_backbone().cuda().eval()
    rec.load_state_dict(torch.load(m68.RUNS / "real_deg_s0.pt"))
    T = json.loads((m68.WORK / "fewshot5d" / "targets.json").read_text(encoding="utf-8"))
    E = embed(rec, [cv2.resize(cv2.imread(p), (160, 160)) for t in T for p in t["enroll"]])
    k, Tm = 0, []
    for t in T:
        Tm.append(E[k:k + len(t["enroll"])].mean(0))
        k += len(t["enroll"])
    Tm = np.stack(Tm)
    Tm /= np.linalg.norm(Tm, axis=1, keepdims=True)
    meta = [json.loads(l) for l in open(SETS["test_swap"] / "meta.jsonl", encoding="utf-8")]
    val = rows_of(det, rec, Tm, SETS["test_swap"], [m for m in meta if m["split"] == "val"])
    res = {}
    for key, root in SETS.items():
        ms = [json.loads(l) for l in open(root / "meta.jsonl", encoding="utf-8")]
        ms = [m for m in ms if m["split"] == "test"]
        res[key] = metrics(val, rows_of(det, rec, Tm, root, ms))
        print(key, json.dumps(res[key], ensure_ascii=False), flush=True)
    allr = json.loads(m68.OUT_JSON.read_text(encoding="utf-8"))
    allr["two_stage"] = res
    m68.OUT_JSON.write_text(json.dumps(allr, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
