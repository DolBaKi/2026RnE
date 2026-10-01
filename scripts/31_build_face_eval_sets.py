"""8차 — 얼굴인식 평가 프로토콜 구성 (학습 신원과 완전히 분리).

1:N 식별 (신원 특정 — 이 과제의 주 지표)
  gallery      eval_vgg 60명 x 등록 사진 5장(원본 얼굴이 가장 큰 = 가장 선명한 등록
               사진) + LFW 5,749명 x 1장 방해 신원  → 총 5,809명 중에서 찾아야 함
  probe_hq            eval_vgg 의 고해상도 얼굴 (face_px >= HQ_MIN)
  probe_lowres_nat    eval_vgg 중 **원래부터** 작게 찍힌 얼굴 (face_px 하위 분위)
  probe_blur_nat      eval_vgg 중 해상도는 충분한데 **원래부터** 흐린 얼굴 (lap_var 하위)
  probe_hq_albudeg    probe_hq 에 Albumentations 열화 — 학습 열화와 같은 계열이라
                      "자기 열화 과적합"을 드러내기 위한 대조군 (5차 교훈)
1:1 검증
  lfw_pairs    LFW 표준 6,000쌍 (10-fold)

자연 저화질 프로브는 열화 함수를 전혀 거치지 않은 실제 사진이다 — 5차에서 확인한
"우리가 만든 열화로 평가하면 개선이 과대평가된다"는 함정을 피하기 위한 핵심 세트.

출력  data/dataset/faces_v1/eval_protocol.json
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rareobj import paths  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

ROOT = paths.DATASET / "faces_v1"
N_ENROLL = 5
HQ_MIN = 100            # 원본 얼굴 박스 한 변(px)
LOWRES_Q = 0.12         # eval_vgg 전체 face_px 하위 12%
BLUR_Q = 0.08           # face_px >= HQ_MIN 중 lap_var 하위 8%
HQ_PER_ID = 25          # probe_hq 는 신원당 최대 25장 (계산량)
SEED = 0


def main() -> int:
    rows = list(csv.DictReader(open(ROOT / "manifest.csv", encoding="utf-8")))
    for r in rows:
        r["face_px"] = float(r["face_px"])
        r["lap_var"] = float(r["lap_var"])
        r["gid"] = int(r["gid"])
    rng = random.Random(SEED)

    vgg = [r for r in rows if r["role"] == "eval_vgg"]
    by_id = defaultdict(list)
    for r in vgg:
        by_id[r["gid"]].append(r)
    gallery, rest = [], []
    for gid, rs in sorted(by_id.items()):
        rs = sorted(rs, key=lambda r: -r["face_px"])
        gallery += [r["path"] for r in rs[:N_ENROLL]]
        rest += rs[N_ENROLL:]

    lowres_thr = float(np.quantile([r["face_px"] for r in vgg], LOWRES_Q))
    hq_pool = [r for r in rest if r["face_px"] >= HQ_MIN]
    blur_thr = float(np.quantile([r["lap_var"] for r in hq_pool], BLUR_Q))
    lowres = [r for r in rest if r["face_px"] <= lowres_thr]
    blur = [r for r in hq_pool if r["lap_var"] <= blur_thr]
    blur_set = {r["path"] for r in blur}
    hq_by_id = defaultdict(list)
    for r in hq_pool:
        if r["path"] not in blur_set:
            hq_by_id[r["gid"]].append(r)
    hq = []
    for gid, rs in sorted(hq_by_id.items()):
        rng.shuffle(rs)
        hq += rs[:HQ_PER_ID]

    lfw = [r for r in rows if r["role"] == "eval_lfw"]
    lfw_first = {}
    lfw_by_orig = {}
    for r in lfw:
        lfw_first.setdefault(r["gid"], r["path"])
        lfw_by_orig[Path(r["orig"]).name] = r["path"]
    distractors = [p for _, p in sorted(lfw_first.items())]

    pairs = []
    with open(paths.RAW / "faces" / "lfw" / "pairs.csv", encoding="utf-8") as f:
        next(f)
        for k, line in enumerate(f):
            t = [x for x in line.strip().split(",") if x != ""]
            if len(t) == 3:
                a, b, same = (t[0], int(t[1])), (t[0], int(t[2])), 1
            else:
                a, b, same = (t[0], int(t[1])), (t[2], int(t[3])), 0
            pa = lfw_by_orig[f"{a[0]}_{a[1]:04d}.jpg"]
            pb = lfw_by_orig[f"{b[0]}_{b[1]:04d}.jpg"]
            pairs.append({"a": pa, "b": pb, "same": same, "fold": k // 600})

    gid_of = {r["path"]: r["gid"] for r in rows}
    proto = {
        "gallery_enroll": gallery,
        "gallery_distractors": distractors,
        "probes": {
            "probe_hq": [r["path"] for r in hq],
            "probe_lowres_nat": [r["path"] for r in lowres],
            "probe_blur_nat": [r["path"] for r in blur],
        },
        "lfw_pairs": pairs,
        "gid_of": {p: gid_of[p] for p in set(gallery) | set(distractors)
                   | {r["path"] for r in hq + lowres + blur}},
        "thresholds": {"lowres_face_px_max": round(lowres_thr, 1), "hq_face_px_min": HQ_MIN,
                       "blur_lap_var_max": round(blur_thr, 1)},
        "stats": {
            "gallery_ids": len(by_id) + len(distractors),
            "enroll_ids": len(by_id), "enroll_imgs": len(gallery), "distractor_ids": len(distractors),
            "probe_hq": len(hq), "probe_lowres_nat": len(lowres), "probe_blur_nat": len(blur),
            "lfw_pairs": len(pairs),
            "lowres_face_px_median": float(np.median([r["face_px"] for r in lowres])),
            "hq_face_px_median": float(np.median([r["face_px"] for r in hq])),
        },
    }
    (ROOT / "eval_protocol.json").write_text(json.dumps(proto, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"thresholds": proto["thresholds"], "stats": proto["stats"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
