"""9차 장면 다양화 — WIDER FACE(사람이 그린 얼굴 박스, 61개 행사 장면)에서 학습 사진 후보를 고른다.

모스크 CCTV 한 종류만으로는 "여러 CCTV 장면에 적용 가능"을 주장할 수 없다(사용자 지적 2026-11-02).
WIDER 는 CC BY-NC-ND(연구용) 이고 박스가 사람 라벨이라 SCRFD 의사 라벨의 순환 평가 문제도 없다.
  학습 사진 후보  WIDER_train 의 군중 장면 범주에서, 타깃 10명 + 모르는 사람을 넣을 자리가 충분한 사진
  자리            사람 박스(가림·자세 0, 무효 아님, 폭 ≥ TARGET_MIN_W) 와 IoU ≥ 0.4 로 맞는 SCRFD 검출
                  (점수 ≥ 0.6, 정면) — 스왑에 필요한 눈·코·입 5점은 SCRFD 에서 가져온다
  val (WIDER_val) 은 학습에 쓰지 않고 시험용으로 남긴다.

출력  data/dataset/CCTV_work/wider_candidates.json, previews/wider_candidates_*.jpg
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rareobj import paths  # noqa: E402
from rareobj.cctv_faceswap import SCRFD  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

WIDER = paths.DATASET / "WIDER"
WORK = paths.DATASET / "CCTV_work"
CROWD = ["0--Parade", "2--Demonstration", "3--Riot", "7--Cheering", "8--Election_Campain", "10--People_Marching",
         "11--Meeting", "12--Group", "17--Ceremony", "18--Concerts", "21--Festival", "23--Shoppers",
         "28--Sports_Fan", "56--Voter", "61--Street_Battle"]
TARGET_MIN_W, MIN_SLOTS, PER_CAT = 20, 14, 3


def read_gt(split: str) -> dict[str, np.ndarray]:
    """wider_face_{split}_bbx_gt.txt → {상대경로: (N,10) x y w h blur expr illum invalid occl pose}"""
    L = (WIDER / "wider_face_split" / f"wider_face_{split}_bbx_gt.txt").read_text().split("\n")
    out, i = {}, 0
    while i < len(L) and L[i]:
        f, n = L[i], int(L[i + 1])
        rows = [list(map(int, L[i + 2 + k].split()[:10])) for k in range(max(n, 1))]
        out[f] = np.array([r for r in rows if r[2] > 0], np.int32).reshape(-1, 10)
        i += 2 + max(n, 1)
    return out


def iou_mat(A, B):
    x0 = np.maximum(A[:, None, 0], B[None, :, 0]); y0 = np.maximum(A[:, None, 1], B[None, :, 1])
    x1 = np.minimum(A[:, None, 2], B[None, :, 2]); y1 = np.minimum(A[:, None, 3], B[None, :, 3])
    i = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    a = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1]); b = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return i / (a[:, None] + b[None] - i + 1e-9)


def match_slots(det, img, gt, scale=2.0):
    """사람 박스마다 SCRFD 검출을 맞춰 5점을 붙인다. 반환: 사람 박스 xyxy, 맞춘 kps(없으면 None), 점수."""
    from importlib import import_module
    is_frontal = import_module("60_cctv_pseudo_labels").is_frontal
    G = np.c_[gt[:, :2], gt[:, :2] + gt[:, 2:4]].astype(np.float32)
    b, s, k = det.detect(img, scale=scale, thr=0.3)
    kps, sc, fr = [None] * len(G), np.zeros(len(G)), np.zeros(len(G), bool)
    if len(b) and len(G):
        M = iou_mat(G, b)
        for gi in range(len(G)):
            j = int(M[gi].argmax())
            if M[gi, j] >= 0.4:
                kps[gi], sc[gi], fr[gi] = k[j].tolist(), float(s[j]), is_frontal(b[j], k[j])
    return G, kps, sc, fr


def main() -> int:
    gt = read_gt("train")
    det = SCRFD()
    cands = []
    for f, g in gt.items():
        if f.split("/")[0] not in CROWD or len(g) < MIN_SLOTS:
            continue
        ok = (g[:, 7] == 0) & (g[:, 8] == 0) & (g[:, 9] == 0) & (g[:, 2] >= TARGET_MIN_W)
        if ok.sum() < MIN_SLOTS:
            continue
        img = cv2.imread(str(WIDER / "WIDER_train" / "images" / f))
        G, kps, sc, fr = match_slots(det, img, g)
        slots = [i for i in range(len(G)) if ok[i] and kps[i] is not None and sc[i] >= 0.6 and fr[i]]
        if len(slots) >= MIN_SLOTS:
            cands.append({"file": f, "cat": f.split("/")[0], "w": img.shape[1], "h": img.shape[0],
                          "n_gt": int(len(g)), "n_slots": len(slots),
                          "med_w": int(np.median(g[:, 2]))})
    print(f"후보 {len(cands)}장", flush=True)
    by = {}
    for c in sorted(cands, key=lambda c: -c["n_gt"]):
        by.setdefault(c["cat"], []).append(c)
    for cat, cs in sorted(by.items()):
        print(f"  {cat:24s} {len(cs):3d}장  예: " + ", ".join(f"{c['n_gt']}명/{c['n_slots']}자리" for c in cs[:3]))
    (WORK / "wider_candidates.json").write_text(json.dumps(cands, ensure_ascii=False, indent=1), encoding="utf-8")

    # 범주마다 사람 수가 많은 순 PER_CAT 장 미리보기
    (WORK / "previews").mkdir(parents=True, exist_ok=True)
    tiles = []
    for cat, cs in sorted(by.items()):
        for c in cs[:PER_CAT]:
            im = cv2.imread(str(WIDER / "WIDER_train" / "images" / c["file"]))
            im = cv2.resize(im, (480, int(480 * im.shape[0] / im.shape[1])))[:360]
            im = cv2.copyMakeBorder(im, 0, 360 - im.shape[0], 0, 0, cv2.BORDER_CONSTANT)
            cv2.putText(im, f"{len(tiles)}: {cat.split('--')[1]} {c['n_gt']}p/{c['n_slots']}s", (6, 22), 0, 0.6,
                        (0, 255, 255), 2)
            c["tile"] = len(tiles)
            tiles.append(im)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    for p in range(0, len(rows), 4):
        cv2.imwrite(str(WORK / "previews" / f"wider_candidates_{p // 4}.jpg"), np.vstack(rows[p:p + 4]))
    (WORK / "wider_candidates.json").write_text(json.dumps(cands, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
