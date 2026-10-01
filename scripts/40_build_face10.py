"""8차(축소판) — 10명 클래스 데이터셋: "CCTV 인파 속에서 이 10명을 찾아라".

480명 전체 합성이 ~10시간 걸려 사용자 지시로 범위를 10명으로 줄였다. 클래스 0~9 =
VGGFace2(원본 train)에서 사진이 가장 많은 인물 중 남녀 5명씩. 나머지 모든 사람은
"모르는 사람"이다 — 실제 CCTV 검색처럼, 10명이 아닌 사람은 아무에게도 매칭되면 안 된다.

  targets          10명 — faces_v1 은 1인당 150장 상한이 있었으므로 원본 전부(600~720장)를 다시 정렬
  unknown_train    VGGFace2 train 의 나머지 470명 x 20장 (faces_v1 재사용) — 학습 시 "모르는 사람"
  unknown_test     eval_vgg 60명 (학습에 한 번도 안 나옴) — 평가용 모르는 사람

타깃 1인당 분할 (사진 단위, 겹치지 않음)
  enroll           원본 얼굴이 가장 큰 정면 사진 5장 — 갤러리 등록 / find_person 질의 사진
  test_lowres_nat  원본 얼굴 박스 ≤ 43px (faces_v1 평가와 같은 기준) — **원래부터** 저해상도
  test_hq          나머지 고해상도(≥100px) 중 15%
  train            나머지 전부 — 생성형 CCTV 합성의 원본 사진도 여기서만 뽑는다

성별은 VGGFace2 메타데이터가 zip에 없어서 CLIP 제로샷(정렬 얼굴 8장 다수결)으로 정했다.

출력  data/dataset/faces10/images/<cls>/*.jpg, manifest.csv, split.json, targets.csv
"""

from __future__ import annotations

import csv
import importlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rareobj import genmodels, paths  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

b30 = importlib.import_module("30_build_face_dataset")
V1 = paths.DATASET / "faces_v1"
OUT = paths.DATASET / "faces10"
VGG = paths.RAW / "faces" / "vggface2"
N_PER_GENDER = 5
N_CAND = 30
N_ENROLL = 5
MIN_ENROLL_SAT = 40.0  # HSV 채도 평균 — 흑백 사진·스케치 제외
LOWRES_MAX = 43.0      # faces_v1 eval_protocol 의 lowres_face_px_max 와 동일
HQ_MIN = 100.0
TEST_HQ_FRAC = 0.15
UNKNOWN_TRAIN_PER_ID = 20
SEED = 0


def main() -> int:
    from facenet_pytorch import MTCNN

    rng = random.Random(SEED)
    v1 = list(csv.DictReader(open(V1 / "manifest.csv", encoding="utf-8")))
    by_name = defaultdict(list)
    for r in v1:
        by_name[r["name"]].append(r)

    # 1) 후보 = 원본 사진 수 상위 30명 → CLIP 성별 판정 → 남녀 5명씩
    counts = sorted(((len(list((VGG / "train" / d).glob("*.jpg"))), d) for d in
                     (p.name for p in (VGG / "train").iterdir())), reverse=True)[:N_CAND]
    clip = genmodels.ClipScorer("cuda")
    labels = ["a photo of a man's face", "a photo of a woman's face"]
    gender = {}
    for n, ident in counts:
        votes = []
        for r in by_name[ident][:8]:
            p = clip.classify(Image.open(V1 / r["path"]).convert("RGB"), labels)
            votes.append(p[labels[0]] > 0.5)
        gender[ident] = "M" if np.mean(votes) >= 0.5 else "F"
    del clip
    men = [d for n, d in counts if gender[d] == "M"][:N_PER_GENDER]
    women = [d for n, d in counts if gender[d] == "F"][:N_PER_GENDER]
    targets = sorted(men + women, key=lambda d: -dict((d2, n) for n, d2 in counts)[d])
    print("targets:", [(d, gender[d], dict((d2, n) for n, d2 in counts)[d]) for d in targets])

    # 2) 10명의 원본 전부 정렬 (faces_v1 과 같은 규칙: MTCNN, 160, margin 14)
    mtcnn = MTCNN(image_size=160, margin=14, keep_all=True, device="cuda")
    rows = []
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    for cls, ident in enumerate(targets):
        d = OUT / "images" / f"{cls:02d}"
        d.mkdir(exist_ok=True)
        for k, f in enumerate(sorted((VGG / "train" / ident).glob("*.jpg"))):
            bgr = cv2.imread(str(f))
            h, w = bgr.shape[:2]
            s = b30.DET_SIDE / max(h, w)
            small = cv2.resize(bgr, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
            boxes, probs, lms = mtcnn.detect(Image.fromarray(small[:, :, ::-1]), landmarks=True)
            box = b30.pick_box(None if boxes is None else boxes / s, probs, w, h)
            if box is None:
                continue
            j = int(np.argmin([np.abs(bb / s - box).sum() for bb in boxes]))
            lm = lms[j] / s
            ed = float(np.hypot(*(lm[0] - lm[1])))
            mid = (lm[0] + lm[1]) / 2
            frontal = abs(lm[2][0] - mid[0]) < 0.12 * ed and abs(lm[0][1] - lm[1][1]) < 0.15 * ed
            face = b30.crop_face(bgr, box)
            sat = float(cv2.cvtColor(face, cv2.COLOR_BGR2HSV)[:, :, 1].mean())
            rel = f"images/{cls:02d}/{k:04d}.jpg"
            cv2.imwrite(str(OUT / rel), face, [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append({"path": rel, "cls": cls, "name": ident, "gender": gender[ident],
                         "orig": f.relative_to(paths.RAW / "faces").as_posix(),
                         "face_px": round(float(min(box[2] - box[0], box[3] - box[1])), 1),
                         "lap_var": round(float(cv2.Laplacian(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY),
                                                              cv2.CV_64F).var()), 1),
                         "frontal": int(frontal), "sat": round(sat, 1)})
        print(f"  cls {cls} {ident}: {sum(r['cls'] == cls for r in rows)} faces", flush=True)

    # 3) 사진 단위 분할
    split = defaultdict(list)
    for cls in range(len(targets)):
        rs = [r for r in rows if r["cls"] == cls]
        # 등록 사진: 정면 + 컬러(채도) — 얼굴 크기만 보고 골랐더니 연필 스케치가 섞였다
        enroll = sorted([r for r in rs if r["frontal"] and r["sat"] >= MIN_ENROLL_SAT],
                        key=lambda r: -r["face_px"])[:N_ENROLL]
        used = {r["path"] for r in enroll}
        low = [r for r in rs if r["face_px"] <= LOWRES_MAX and r["path"] not in used]
        used |= {r["path"] for r in low}
        hq = [r for r in rs if r["face_px"] >= HQ_MIN and r["path"] not in used]
        rng.shuffle(hq)
        thq = hq[:int(len(hq) * TEST_HQ_FRAC)]
        used |= {r["path"] for r in thq}
        for name, lst in (("enroll", enroll), ("test_lowres_nat", low), ("test_hq", thq),
                          ("train", [r for r in rs if r["path"] not in used])):
            split[name] += [r["path"] for r in lst]
    for r in rows:
        r["split"] = next(k for k, v in split.items() if r["path"] in set(v))

    # 4) 모르는 사람 풀 (faces_v1 정렬본 재사용, 경로는 faces_v1 기준)
    tset = set(targets)
    unk_tr = defaultdict(list)
    for r in v1:
        if r["role"] == "train" and r["name"] not in tset:
            unk_tr[r["name"]].append(r["path"])
    split["unknown_train"] = [p for ps in unk_tr.values() for p in rng.sample(ps, min(UNKNOWN_TRAIN_PER_ID, len(ps)))]
    ev = json.loads((V1 / "eval_protocol.json").read_text(encoding="utf-8"))
    split["unknown_test_hq"] = ev["probes"]["probe_hq"]
    split["unknown_test_lowres_nat"] = ev["probes"]["probe_lowres_nat"]

    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    with open(OUT / "targets.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["cls", "vggface2_id", "gender", "n_faces", "n_train", "n_test_hq", "n_test_lowres_nat"])
        for cls, ident in enumerate(targets):
            rs = [r for r in rows if r["cls"] == cls]
            wr.writerow([cls, ident, gender[ident], len(rs)] +
                        [sum(r["split"] == s for r in rs) for s in ("train", "test_hq", "test_lowres_nat")])
    (OUT / "split.json").write_text(json.dumps(split, ensure_ascii=False), encoding="utf-8")
    print({k: len(v) for k, v in split.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
