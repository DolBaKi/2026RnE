"""9차 — 실제 CCTV 군중 프레임(data/dataset/CCTV/RawDataFrames)에 SCRFD-10G 로 얼굴 의사 라벨.

데이터에 박스 라벨이 없다. MTCNN 은 고개 숙인·작은 얼굴을 많이 놓쳐(샘플 프레임 14개 vs SCRFD 67개)
SCRFD 를 2배 확대로 돌린다. 라벨 임계값은 0.5(그 아래는 머리 꼭대기 등 오검출이 섞인다),
0.3~0.5 는 학습에서 무시 영역으로 쓸 수 있게 점수와 함께 남긴다.

카메라 분할 — "적은 장면으로 학습해 처음 보는 카메라에서 통하는가":
  video_summary.png 의 클립 34개(카메라 24대) 시작 프레임을 눈으로 읽고(CLIPS), ±30 프레임 안에서
  HSV 히스토그램 차이가 가장 큰 곳으로 맞춘다.
  train 5대 / val 2대 / test 나머지 17대.

출력  data/dataset/CCTV/pseudo_labels.jsonl  (프레임마다 clip, cam, split, boxes, scores, kps, frontal)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rareobj import paths  # noqa: E402
from rareobj.cctv_faceswap import SCRFD  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

ROOT = paths.DATASET / "CCTV"
OUT = ROOT / "pseudo_labels.jsonl"
# (클립 번호, 대략 시작 프레임(0부터), 카메라) — video_summary.png 에서 읽음
CLIPS = [(1, 0, 5), (2, 16, 1), (3, 62, 2), (4, 229, 3), (5, 370, 4), (6, 566, 5), (7, 624, 6), (8, 760, 5),
         (9, 932, 7), (10, 1040, 8), (11, 1125, 9), (12, 1270, 10), (13, 1352, 5), (14, 1386, 11),
         (15, 1434, 12), (16, 1512, 12), (17, 1656, 3), (18, 1759, 5), (19, 1794, 13), (20, 1858, 5),
         (21, 1915, 14), (22, 1940, 15), (23, 2022, 16), (24, 2080, 3), (25, 2136, 13), (26, 2208, 17),
         (27, 2388, 18), (28, 2945, 19), (29, 3225, 20), (30, 3417, 21), (31, 3915, 6), (32, 4115, 22),
         (33, 4489, 23), (34, 4586, 24)]
TRAIN_CAMS, VAL_CAMS = {2, 4, 12, 18, 21}, {19, 22}
SEARCH = 30
DET_SCALE, DET_THR = 2.0, 0.3
FRONTAL_EYE_RATIO = 0.28      # 눈 사이 거리 / 박스 폭 — 고개 숙이거나 옆을 보면 작아진다


def frame_hist(f):
    im = cv2.cvtColor(cv2.resize(cv2.imread(str(f)), (240, 120), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([im], [0, 1], None, [30, 16], [0, 180, 0, 256])
    return cv2.normalize(h, h)


def clip_starts(files) -> list[int]:
    H = [frame_hist(f) for f in files]
    d = np.array([0.0] + [cv2.compareHist(H[i - 1], H[i], cv2.HISTCMP_BHATTACHARYYA) for i in range(1, len(H))])
    starts = [0]
    for _, s0, _ in CLIPS[1:]:
        lo, hi = max(starts[-1] + 1, s0 - SEARCH), min(len(files) - 1, s0 + SEARCH)
        starts.append(int(lo + np.argmax(d[lo:hi + 1])))
    return starts


def is_frontal(box, kps) -> bool:
    le, re, nose = kps[0], kps[1], kps[2]
    ed = float(np.hypot(*(le - re)))
    lo, hi = sorted((le[0], re[0]))
    return bool(ed / max(box[2] - box[0], 1e-3) >= FRONTAL_EYE_RATIO and lo <= nose[0] <= hi and nose[1] > max(le[1], re[1]))


def main() -> int:
    files = sorted((ROOT / "RawDataFrames").glob("*.png"))
    starts = clip_starts(files)
    print("클립 시작:", starts, flush=True)
    det = SCRFD()
    t0 = time.time()
    ci = 0
    with open(OUT, "w", encoding="utf-8") as f:
        for i, fp in enumerate(files):
            while ci + 1 < len(starts) and i >= starts[ci + 1]:
                ci += 1
            clip, _, cam = CLIPS[ci]
            split = "train" if cam in TRAIN_CAMS else "val" if cam in VAL_CAMS else "test"
            img = cv2.imread(str(fp))
            b, s, k = det.detect(img, scale=DET_SCALE, thr=DET_THR)
            rec = {"file": fp.name, "idx": i, "clip": clip, "cam": cam, "split": split, "w": img.shape[1],
                   "h": img.shape[0], "boxes": np.round(b, 2).tolist(), "scores": np.round(s, 3).tolist(),
                   "kps": np.round(k, 2).tolist(), "frontal": [is_frontal(bb, kk) for bb, kk in zip(b, k)]}
            f.write(json.dumps(rec) + "\n")
            if i % 250 == 0:
                print(f"  {i}/{len(files)} clip {clip} cam {cam} faces {int((s >= 0.5).sum())}  {time.time() - t0:.0f}s",
                      flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
