"""8차 — 인파 테스트 프레임에서 얼굴 검출 + 정답 매칭 (모델 무관, 한 번만 실행).

"찾을 사람 사진 → CCTV 인파 속에서 그 사람 찾기"를 실제 시스템처럼 평가하려면
정답 박스를 그대로 쓰면 안 되고, 프레임에서 얼굴을 **직접 검출**해야 한다 —
검출기가 놓치거나 어긋나게 잡은 얼굴도 시스템 성능의 일부다.

각 검출을 정답과 IoU로 매칭해
  gid >= 0   우리가 넣은 eval_vgg 인물 (찾아야 하는 사람)
  gid == -1  생성 인물 / 되돌린 인물 / 군중 속 다른 사람 (모르는 사람)
으로 표시한다. 정답 중 검출되지 않은 얼굴은 miss 로 따로 센다.

입력  data/dataset/<data>/cctv_frames/test/{images,frames.jsonl}    (--data faces_v1 | faces10)
출력  data/dataset/<data>/cctv_frames/test/dets/*.jpg   (160 정렬 크롭)
      data/dataset/<data>/cctv_frames/test/dets.json
faces10 은 정답 라벨이 cls(0~9 타깃, 10 = 삽입한 모르는 사람)라 10 은 -1 로 바꿔 쓴다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rareobj import paths  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

OUT_SIZE, MARGIN = 160, 14
DET_PROB = 0.90
IOU_MATCH = 0.30


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def crop_box(frame, box):
    H, W = frame.shape[:2]
    bw, bh = box[2] - box[0], box[3] - box[1]
    mx, my = MARGIN * bw / (OUT_SIZE - MARGIN), MARGIN * bh / (OUT_SIZE - MARGIN)
    x0, y0 = int(max(box[0] - mx / 2, 0)), int(max(box[1] - my / 2, 0))
    x1, y1 = int(min(box[2] + mx / 2, W)), int(min(box[3] + my / 2, H))
    return cv2.resize(frame[y0:y1, x0:x1], (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_CUBIC)


def label_of(face: dict) -> int:
    if "gid" in face:
        return face["gid"]
    return face["cls"] if face["cls"] < 10 else -1


def main() -> int:
    from facenet_pytorch import MTCNN

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="faces_v1")
    args = ap.parse_args()
    D = paths.DATASET / args.data / "cctv_frames" / "test"

    mtcnn = MTCNN(keep_all=True, min_face_size=14, device="cuda")
    (D / "dets").mkdir(exist_ok=True)
    frames = [json.loads(l) for l in open(D / "frames.jsonl", encoding="utf-8")]
    dets, n_gt, n_hit = [], 0, 0
    for fi, fr in enumerate(frames):
        img = cv2.imread(str(D / fr["image"]))
        boxes, probs = mtcnn.detect(Image.fromarray(img[:, :, ::-1]))
        boxes = [] if boxes is None else [b for b, p in zip(boxes, probs) if p >= DET_PROB]
        gts = fr["faces"]
        n_gt += sum(label_of(g) >= 0 for g in gts)
        used = set()
        for k, b in enumerate(boxes):
            best, bi = 0.0, -1
            for j, g in enumerate(gts):
                v = iou(b, g["box"])
                if v > best and j not in used:
                    best, bi = v, j
            gid = -1
            if best >= IOU_MATCH:
                used.add(bi)
                gid = label_of(gts[bi])
            name = f"{Path(fr['image']).stem}_{k}.jpg"
            cv2.imwrite(str(D / "dets" / name), crop_box(img, b), [cv2.IMWRITE_JPEG_QUALITY, 95])
            dets.append({"frame": fi, "crop": f"cctv_frames/test/dets/{name}", "gid": gid,
                         "box": [round(float(v), 1) for v in b], "face_px": round(float(b[2] - b[0]), 1)})
        n_hit += sum(label_of(gts[j]) >= 0 for j in used)
    out = {"n_frames": len(frames), "n_gt_targets": n_gt, "n_detected_targets": n_hit,
           "detection_recall": n_hit / max(n_gt, 1), "n_dets": len(dets),
           "n_unknown_dets": sum(d["gid"] < 0 for d in dets), "dets": dets}
    (D / "dets.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print({k: v for k, v in out.items() if k != "dets"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
