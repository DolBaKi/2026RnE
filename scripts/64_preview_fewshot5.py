"""9차 — fewshot5 데이터 눈 확인용 시트 (실존 인물이라 data/ 아래에만 저장).

  photos_v00.jpg       사진 5장의 첫 변형본 — 노란 박스·번호 = 타깃(T0~T9), 하늘색 = 스왑한 모르는 사람
  swap_targets_<사진>.jpg  그 사진에 넣은 타깃 10명: [원본 | 스왑 후 | 넣은 사람 사진]
  targets.jpg          타깃 10명 등록 사진
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rareobj import paths  # noqa: E402

OUT = paths.DATASET / "CCTV_work" / (sys.argv[1] if len(sys.argv) > 1 else "fewshot5")    # fewshot5d = 장면 다양화판
RAW = {"cctv": paths.DATASET / "CCTV" / "RawDataFrames", "wider": paths.DATASET / "WIDER" / "WIDER_train" / "images"}
T = 140


def main() -> int:
    meta = [json.loads(l) for l in open(OUT / "meta.jsonl", encoding="utf-8")]
    tg = json.loads((OUT / "targets.json").read_text(encoding="utf-8"))
    v0 = [m for m in meta if m["split"] == "train" and m["image"].endswith("_v00.jpg")]
    frames = []
    for m in v0:
        im = cv2.imread(str(OUT / m["image"]))
        for f in m["swaps"]:
            b = np.int32(f["box"])
            col = (0, 255, 255) if f["cls"] < 10 else (255, 200, 0)
            cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), col, 2 if f["cls"] < 10 else 1)
            if f["cls"] < 10:
                cv2.putText(im, f"T{f['cls']}", (b[0], b[1] - 3), 0, 0.5, col, 2)
        cv2.putText(im, f"{m['cam']} {Path(m['src_frame']).stem[-24:]}", (8, 24), 0, 0.8, (255, 255, 255), 2)
        frames.append(cv2.resize(im, (1024, int(1024 * im.shape[0] / im.shape[1]))))
    H = max(f.shape[0] for f in frames)
    frames = [cv2.copyMakeBorder(f, 0, H - f.shape[0], 0, 0, cv2.BORDER_CONSTANT) for f in frames]
    frames.append(np.zeros_like(frames[0]))
    cv2.imwrite(str(OUT / "previews" / "photos_v00.jpg"),
                np.vstack([np.hstack(frames[i:i + 2]) for i in range(0, 6, 2)]))

    z = lambda im: cv2.resize(im, (T, T), interpolation=cv2.INTER_CUBIC)
    for m in v0:
        orig = cv2.imread(str(RAW[m.get("src", "cctv")] / m["src_frame"]))
        sw = cv2.imread(str(OUT / m["image"]))
        rows = []
        for f in sorted([f for f in m["swaps"] if f["cls"] < 10], key=lambda f: f["cls"]):
            x0, y0, x1, y1 = np.int32(f["box"])
            mg = int((x1 - x0) * 0.6)
            sl = (slice(max(0, y0 - mg), y1 + mg), slice(max(0, x0 - mg), x1 + mg))
            src = cv2.imread(tg[f["cls"]]["src_train"][0])
            t = cv2.copyMakeBorder(np.hstack([z(orig[sl]), z(sw[sl]), z(src)]), 20, 3, 3, 3,
                                   cv2.BORDER_CONSTANT, value=(40, 40, 40))
            age = f"  age {f['slot_age']:.0f}/{tg[f['cls']]['age']:.0f}" if f.get("slot_age") is not None else ""
            cv2.putText(t, f"T{f['cls']} cos {f['id_cos']:.2f} {x1 - x0}px{age}", (5, 15), 0, 0.45,
                        (0, 255, 255), 1)
            rows.append(t)
        head = np.full((24, rows[0].shape[1], 3), 40, np.uint8)
        for k, s in enumerate(["original", "swapped", "source photo"]):
            cv2.putText(head, s, (8 + k * T, 17), 0, 0.5, (255, 255, 255), 1)
        cols = [np.vstack([head] + rows[i::2]) for i in range(2)]
        h = max(c.shape[0] for c in cols)
        cols = [cv2.copyMakeBorder(c, 0, h - c.shape[0], 0, 0, cv2.BORDER_CONSTANT) for c in cols]
        cv2.imwrite(str(OUT / "previews" / f"swap_targets_{Path(m['src_frame']).stem}.jpg"),
                    np.hstack([cols[0], np.zeros((h, 8, 3), np.uint8), cols[1]]))

    gal = []
    for t in tg:
        im = z(cv2.imread(t["enroll"][0]))
        cv2.putText(im, f"T{t['cls']} {t['name']}", (4, 14), 0, 0.42, (0, 255, 255), 1)
        gal.append(im)
    cv2.imwrite(str(OUT / "previews" / "targets.jpg"), np.vstack([np.hstack(gal[:5]), np.hstack(gal[5:])]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
