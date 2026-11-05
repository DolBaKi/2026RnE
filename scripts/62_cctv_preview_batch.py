"""9차 — CCTVFaceDataset 을 DataLoader 로 실제로 뽑아 증강·박스가 맞는지 눈으로 확인한다.

출력  data/dataset/CCTV_work/previews/batch_{split}.jpg   (실존 인물 영상이라 reports/ 에 두지 않는다)
      초록 = 얼굴 박스, 왼쪽 위 노란 글씨 = 카메라·스왑 여부
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rareobj import paths  # noqa: E402
from rareobj.cctv_data import CCTVFaceDataset, collate  # noqa: E402

WORK = paths.DATASET / "CCTV_work"


def source_photo(name: str) -> np.ndarray:
    """스왑 소스 신원의 대표 사진 — scripts/61 build_sources 와 같은 풀(faces_v1 학습 신원 / faces10 타깃)."""
    import csv

    for root in (paths.DATASET / "faces_v1", paths.DATASET / "faces10"):
        for r in csv.DictReader(open(root / "manifest.csv", encoding="utf-8")):
            if r["name"] == name:
                return cv2.imread(str(root / r["path"]))
    return np.zeros((160, 160, 3), np.uint8)


def swap_source_sheet(n: int = 12, seed: int = 0):
    """한 줄 = [원본 얼굴 | 스왑 후 | 넣은 사람 사진] — 어떤 사진으로 바꿨는지 같이 본다."""
    import json
    import random

    meta = [json.loads(l) for l in open(WORK / "yolo" / "meta.jsonl", encoding="utf-8")]
    sw = [(m, s) for m in meta if m["swaps"] for s in m["swaps"]]
    random.Random(seed).shuffle(sw)
    # 큰 얼굴 위주로 — 19px 얼굴은 확대해도 차이가 안 보인다
    sw = sorted(sw[:200], key=lambda x: -(x[1]["box"][2] - x[1]["box"][0]))[:n]
    T = 150
    z = lambda im: cv2.resize(im, (T, T), interpolation=cv2.INTER_CUBIC)
    rows = []
    for m, s in sw:
        orig = cv2.imread(str(paths.DATASET / "CCTV" / "RawDataFrames" / m["src_frame"]))
        swp = cv2.imread(str(WORK / "yolo" / m["image"]))
        x0, y0, x1, y1 = np.int32(s["box"])
        mg = int((x1 - x0) * 0.6)
        sl = (slice(max(0, y0 - mg), y1 + mg), slice(max(0, x0 - mg), x1 + mg))
        tile = np.hstack([z(orig[sl]), z(swp[sl]), z(source_photo(s["src"]))])
        tile = cv2.copyMakeBorder(tile, 22, 4, 4, 4, cv2.BORDER_CONSTANT, value=(40, 40, 40))
        tag = f"{s['src']} cls{s['cls']} cos {s['id_cos_after']:.2f}  {x1 - x0}px"
        cv2.putText(tile, tag, (6, 16), 0, 0.45, (0, 255, 255), 1)
        rows.append(tile)
    head = np.full((26, rows[0].shape[1], 3), 40, np.uint8)
    for k, t in enumerate(["original", "swapped", "source photo"]):
        cv2.putText(head, t, (10 + k * T, 18), 0, 0.5, (255, 255, 255), 1)
    cols = [np.vstack([head] + rows[i::2]) for i in range(2)]
    cv2.imwrite(str(WORK / "previews" / "swap_with_source.jpg"),
                np.hstack([cols[0], np.full((cols[0].shape[0], 10, 3), 0, np.uint8), cols[1]]))


def main() -> int:
    swap_source_sheet()
    torch.manual_seed(0)
    for split in ("train", "val"):
        ds = CCTVFaceDataset(WORK / "yolo", split)
        dl = torch.utils.data.DataLoader(ds, batch_size=6, shuffle=True, collate_fn=collate, num_workers=0)
        x, t = next(iter(dl))
        print(split, "샘플", len(ds), "배치", tuple(x.shape), "박스 수", [len(v["boxes"]) for v in t])
        tiles = []
        for im, tt in zip(x, t):
            v = np.ascontiguousarray((im.permute(1, 2, 0).numpy() * 255).astype(np.uint8)[:, :, ::-1])
            for b in tt["boxes"].numpy().astype(int):
                cv2.rectangle(v, tuple(b[:2]), tuple(b[2:]), (0, 255, 0), 1)
            cv2.putText(v, f"cam{tt['cam']:02d} {'SWAP' if tt['swapped'] else 'orig'}", (8, 28), 0, 0.9,
                        (0, 255, 255), 2)
            tiles.append(v)
        grid = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
        cv2.imwrite(str(WORK / "previews" / f"batch_{split}.jpg"), grid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
