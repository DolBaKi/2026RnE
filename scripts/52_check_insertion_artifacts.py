"""합성 흔적 판별 검사 — "그려 넣은 얼굴"과 "원래 장면 얼굴"을 분류기가 구분할 수 있는가.

사용자 지적(2026-10-29): "흰색 경계만 학습하는 양상". 인식 모델이 신원 대신 삽입 흔적(선명도·
테두리·색)을 단서로 쓸 수 있다면, 같은 단서를 노린 분류기가 두 집단을 쉽게 가른다. 분류기가
못 가르면(AUC≈0.5) 인식 모델도 그 지름길을 쓸 수 없다.

  양성  frames.jsonl 의 faces (그려 넣은 얼굴)
  음성  같은 프레임에서 그려 넣은 얼굴과 겹치지 않는 원래 얼굴
  둘 다 최종 프레임(CCTV 열화 후)에서 MTCNN 으로 다시 검출해 같은 방식으로 자른다.
  --neg-frontal 이면 음성을 정면 슬롯(같은 필터 통과)이지만 다시 그리지 않은 얼굴로 제한한다 —
  아무 얼굴이나 음성으로 쓰면 "정면·반듯한 얼굴인가"로도 갈려 흔적과 섞인다.
  얼굴 크기 차이만으로 갈리는 착시를 막으려고 양성 크기 5~95% 범위의 음성만 쓰고,
  "크기만 보는" 분류기 AUC 를 같이 보고한다.

  분류기  (a) 수작업 특징(선명도·테두리 대비·색·밝기 통계) 로지스틱 회귀
          (b) ResNet18(ImageNet) 미세조정 64px
          프레임 단위 5-fold — 같은 프레임의 얼굴이 학습·평가에 갈리지 않게.

사용  python scripts/52_check_insertion_artifacts.py --frames data/dataset/faces10/cctv_frames/train --name v3
출력  reports/insertion_artifacts_<name>.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rareobj import paths  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

c35 = importlib.import_module("35_build_crowd_test_dets")    # iou, crop_box(160 정렬 크롭과 같은 여백)
DET_PROB = 0.90
IOU_POS, IOU_NEG = 0.3, 0.1
CROP = 64
N_FOLD = 5


def collect(frames_dir: Path, max_pos: int, seed: int, neg_frontal: bool = False):
    from facenet_pytorch import MTCNN

    mtcnn = MTCNN(keep_all=True, device="cuda")
    frames = [json.loads(l) for l in open(frames_dir / "frames.jsonl", encoding="utf-8")]
    random.Random(seed).shuffle(frames)
    X, y, g, px = [], [], [], []
    n_pos = 0
    for fi, fr in enumerate(frames):
        if max_pos and n_pos >= max_pos:
            break
        img = cv2.imread(str(frames_dir / fr["image"]))
        boxes, probs = mtcnn.detect(Image.fromarray(img[:, :, ::-1]))
        if boxes is None:
            continue
        ins = [f["box"] for f in fr["faces"]]
        unt = fr.get("untouched_frontal", [])
        for b, p in zip(boxes, probs):
            if p < DET_PROB:
                continue
            ov = max((c35.iou(b, t) for t in ins), default=0.0)
            lab = 1 if ov >= IOU_POS else 0 if ov < IOU_NEG else None
            # 정면 대조군: 같은 정면 슬롯 필터를 통과했지만 다시 그리지 않은 얼굴만 음성으로
            if lab == 0 and neg_frontal and max((c35.iou(b, t) for t in unt), default=0.0) < IOU_POS:
                lab = None
            if lab is None:
                continue
            X.append(cv2.resize(c35.crop_box(img, b), (CROP, CROP), interpolation=cv2.INTER_AREA))
            y.append(lab)
            g.append(fi)
            px.append(float(b[2] - b[0]))
            n_pos += lab
    y, g, px = np.array(y), np.array(g), np.array(px)
    lo, hi = np.quantile(px[y == 1], [0.05, 0.95])
    keep = (y == 1) | ((px >= lo) & (px <= hi))
    return np.stack(X)[keep], y[keep], g[keep], px[keep]


def hand_features(crops: np.ndarray) -> np.ndarray:
    """흰 테두리·선명도 차이·색 차이를 직접 겨냥한 특징 — 얼굴 중심부 vs 둘레 링."""
    out = []
    yy, xx = np.mgrid[0:CROP, 0:CROP]
    r = np.hypot(yy - CROP / 2, xx - CROP / 2) / (CROP / 2)
    core, ring = r < 0.45, (r > 0.6) & (r < 0.85)
    for c in crops:
        lab = cv2.cvtColor(c, cv2.COLOR_BGR2LAB).astype(np.float32)
        lap = np.abs(cv2.Laplacian(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), cv2.CV_32F))
        f = []
        for ch in range(3):
            f += [lab[..., ch][core].mean(), lab[..., ch][ring].mean(), lab[..., ch][core].std(), lab[..., ch][ring].std()]
        f += [lap[core].mean(), lap[ring].mean(), lap[core].mean() / (lap[ring].mean() + 1e-3),
              lab[..., 0][core].mean() - lab[..., 0][ring].mean()]
        out.append(f)
    return np.array(out, np.float32)


def auc(score, y):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, score))


def folds(g):
    ug = np.unique(g)
    rng = np.random.default_rng(0)
    rng.shuffle(ug)
    return [np.isin(g, part) for part in np.array_split(ug, N_FOLD)]


def cv_logistic(F_, y, g):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    s = np.zeros(len(y))
    for te in folds(g):
        sc = StandardScaler().fit(F_[~te])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(F_[~te]), y[~te])
        s[te] = clf.predict_proba(sc.transform(F_[te]))[:, 1]
    return auc(s, y)


def cv_cnn(X, y, g, epochs=8):
    import torchvision

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
    xt = torch.from_numpy(X[:, :, :, ::-1].copy()).permute(0, 3, 1, 2).float().div(255)
    s = np.zeros(len(y))
    for k, te in enumerate(folds(g)):
        torch.manual_seed(k)
        net = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        net.fc = nn.Linear(512, 1)
        net = net.cuda()
        opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
        tr = np.where(~te)[0]
        pw = torch.tensor([(y[tr] == 0).sum() / max(1, (y[tr] == 1).sum())]).cuda()
        lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
        for _ in range(epochs):
            net.train()
            np.random.shuffle(tr)
            for i in range(0, len(tr), 64):
                idx = tr[i:i + 64]
                xb = xt[idx].cuda()
                if random.random() < 0.5:
                    xb = xb.flip(3)
                loss = lossf(net((xb - mean) / std).squeeze(1), torch.from_numpy(y[idx]).float().cuda())
                opt.zero_grad()
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            s[te] = torch.sigmoid(net((xt[te].cuda() - mean) / std).squeeze(1)).cpu().numpy()
    return auc(s, y)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="frames.jsonl 과 images/ 가 있는 폴더")
    ap.add_argument("--name", required=True)
    ap.add_argument("--max-pos", type=int, default=0, help="양성 수 상한 (데이터 양을 맞춰 비교할 때)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--neg-frontal", action="store_true",
                    help="음성을 '정면 슬롯이지만 다시 그리지 않은 얼굴'로 제한 (자세·품질 차이 배제)")
    args = ap.parse_args()
    X, y, g, px = collect(Path(args.frames), args.max_pos, args.seed, args.neg_frontal)
    print(f"[{args.name}] 양성(그려 넣음) {int(y.sum())}  음성(원래 얼굴) {int((y == 0).sum())}  프레임 {len(np.unique(g))}")
    res = {"name": args.name, "frames": args.frames, "n_pos": int(y.sum()), "n_neg": int((y == 0).sum()),
           "n_frames": int(len(np.unique(g))),
           "face_px_med_pos": float(np.median(px[y == 1])), "face_px_med_neg": float(np.median(px[y == 0])),
           "auc_size_only": cv_logistic(px[:, None], y, g),
           "auc_handcrafted": cv_logistic(hand_features(X), y, g),
           "auc_cnn": cv_cnn(X, y, g)}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    (paths.REPORTS / f"insertion_artifacts_{args.name}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
