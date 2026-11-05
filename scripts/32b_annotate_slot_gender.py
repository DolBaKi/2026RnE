"""8차 — 인파 장면의 정면 인물(슬롯)마다 성별 판정 (CLIP 제로샷).

사용자 요청: 이미지 생성 시 남자 사진을 합성할지 여자 사진을 합성할지 구분할 것.
신원 조건부 인페인팅은 머리~어깨만 다시 그리고 몸·옷은 원래 생성 인물을 그대로 쓰므로,
남자 타깃을 여자 인물 자리에 넣으면 체형·옷차림과 얼굴이 어긋난다. 슬롯마다 상반신
크롭(머리 위 0.6 얼굴 ~ 아래 4 얼굴, 폭 3 얼굴)을 CLIP 으로 판정해 manifest 에 기록하고,
scripts/41 이 같은 성별의 사람만 그 자리에 넣는다. 확신이 낮은(|p-0.5| < MARGIN) 슬롯은
"?" 로 두고 타깃에는 쓰지 않는다.

입출력  data/interim/generated/face_cctv_scenes_manifest.jsonl  (slots[*].gender, gender_p 추가)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rareobj import genmodels, paths  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

GEN = paths.INTERIM / "generated"
MAN = GEN / "face_cctv_scenes_manifest.jsonl"
LABELS = ["a photo of a man", "a photo of a woman"]
MARGIN = 0.15
FACE_PER_EYE = 2.5


def main() -> int:
    clip = genmodels.ClipScorer("cuda")
    scenes = [json.loads(l) for l in open(MAN, encoding="utf-8")]
    stats = {"M": 0, "F": 0, "?": 0}
    for sc in scenes:
        img = cv2.imread(str(GEN / "face_cctv_scenes" / sc["file"]))
        H, W = img.shape[:2]
        for s in sc["slots"]:
            fs = FACE_PER_EYE * s["eye_dist"]
            cx = (s["leye"][0] + s["reye"][0]) / 2
            cy = (s["leye"][1] + s["reye"][1]) / 2
            x0, x1 = int(max(cx - 1.5 * fs, 0)), int(min(cx + 1.5 * fs, W))
            y0, y1 = int(max(cy - 0.6 * fs, 0)), int(min(cy + 4.0 * fs, H))
            crop = cv2.resize(img[y0:y1, x0:x1], None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            p = clip.classify(Image.fromarray(crop[:, :, ::-1]), LABELS)[LABELS[0]]
            s["gender_p_male"] = round(p, 3)
            s["gender"] = "M" if p >= 0.5 + MARGIN else ("F" if p <= 0.5 - MARGIN else "?")
            stats[s["gender"]] += 1
    with open(MAN, "w", encoding="utf-8") as f:
        for sc in scenes:
            f.write(json.dumps(sc, ensure_ascii=False) + "\n")
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
