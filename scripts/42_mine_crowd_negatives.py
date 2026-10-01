"""8차(10명) — 학습 프레임의 배경 군중 얼굴을 "모르는 사람"으로 채굴 (하드 네거티브).

1차 학습 후 분석: 인파 테스트에서 FAR 1% 임계값을 끌어올린 건 우리가 그려 넣은 모르는
사람(삽입 인물, q99 0.48)이 아니라 장면에 원래 있던 SDXL 생성 군중 얼굴(q99 0.69,
real+synth s1)이었다. 학습의 "모르는 사람" 클래스에는 실존 인물 사진과 삽입 인물만
있었고 배경 군중은 한 번도 없었기 때문 — 모델이 군중 얼굴을 타깃 근처에 놓았다.

scripts/41 train 프레임(테스트와 다른 장면)에서 MTCNN으로 모든 얼굴을 검출하고, 삽입한
인물(타깃·모르는 사람)과 IoU 0.1 이상 겹치지 않는 얼굴을 cls 10 으로 synth_manifest 에
추가한다. 실제 CCTV 시스템에서 현장 영상의 불특정 다수를 네거티브로 쓰는 것과 같다.

출력  data/dataset/faces10/synth_cctv/10/crowd_*.jpg, synth_manifest.csv 에 행 추가(who=crowd)
"""

from __future__ import annotations

import csv
import importlib
import json
import sys
from pathlib import Path

import cv2
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rareobj import paths  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

c35 = importlib.import_module("35_build_crowd_test_dets")
F10 = paths.DATASET / "faces10"
D = F10 / "cctv_frames" / "train"
MAN = F10 / "synth_manifest.csv"
IOU_EXCLUDE = 0.10


def main() -> int:
    from facenet_pytorch import MTCNN

    mtcnn = MTCNN(keep_all=True, min_face_size=14, device="cuda")
    rows = list(csv.DictReader(open(MAN, encoding="utf-8")))
    rows = [r for r in rows if r["who"] != "crowd"]          # 재실행 시 중복 방지
    out_dir = F10 / "synth_cctv" / "10"
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.glob("crowd_*.jpg"):
        p.unlink()
    frames = [json.loads(l) for l in open(D / "frames.jsonl", encoding="utf-8")]
    n = 0
    for fr in frames:
        img = cv2.imread(str(D / fr["image"]))
        boxes, probs = mtcnn.detect(Image.fromarray(img[:, :, ::-1]))
        if boxes is None:
            continue
        for k, (b, p) in enumerate(zip(boxes, probs)):
            if p < c35.DET_PROB or any(c35.iou(b, f["box"]) >= IOU_EXCLUDE for f in fr["faces"]):
                continue
            rel = Path("synth_cctv") / "10" / f"crowd_{Path(fr['image']).stem}_{k}.jpg"
            cv2.imwrite(str(F10 / rel), c35.crop_box(img, b), [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append({"path": rel.as_posix(), "cls": 10, "who": "crowd", "frame": f"cctv_frames/train/{fr['image']}",
                         "face_px": round(float(b[2] - b[0]), 1), "id_cos": ""})
            n += 1
    with open(MAN, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"crowd negatives {n} from {len(frames)} train frames; manifest rows {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
