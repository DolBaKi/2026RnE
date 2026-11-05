"""v4 — 인파 CCTV 장면 생성 (RealVisXL) + 정면 인물 슬롯 (YOLO26-pose + MTCNN 확인 + CLIP 성별).

v3(scripts/32)의 장면은 SDXL-Turbo 로 그렸다. Turbo 는 CFG 가 없어 negative prompt 가 안 듣고
"군중"을 늘 보케·모션블러로 그렸다 — 원래 군중 얼굴은 덩어리, 그려 넣은 얼굴만 선명해서
"선명한 얼굴 = 넣은 사람" 이라는 지름길이 생겼다(2026-10-29 진단, docs/07). v4 는 사실풍 SDXL
파인튜닝(RealVisXL V5) + negative prompt 로 실제 CCTV 처럼 프레임 전체가 고르게 선명한 장면을
그린다. 같은 시드 비교에서 라플라시안 분산 37~115 → 690~1120, 1152x768 장당 ~16초.

슬롯 = 카메라를 향한 인물: pose 키포인트(코+양눈, 코가 두 눈 사이) 통과 후, 원본 얼굴이 MTCNN
으로 확실히(p≥0.95) 검출돼야 한다 — v4 시험에서 pose 만으로는 고개 숙인 사람이 섞여 그 자리에
그린 얼굴이 어색했다. 성별은 상반신 크롭 CLIP 판정(scripts/32b 와 같은 방식).

출력
  data/interim/generated/face_cctv_scenes_v4/{split}_{idx:04d}.jpg
  data/interim/generated/face_cctv_scenes_v4_manifest.jsonl
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rareobj import genmodels, paths  # noqa: E402
from rareobj.faceid_inpaint_xl import load_scene_pipe  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

GEN = paths.INTERIM / "generated"
OUT_DIR = GEN / "face_cctv_scenes_v4"
MANIFEST = GEN / "face_cctv_scenes_v4_manifest.jsonl"
W, H = 1152, 768
STEPS, CFG = 25, 5.0

SCENES = [
    "a crowd of commuters walking toward the camera through a subway station exit",
    "a crowd of pedestrians walking toward the camera on a busy city sidewalk",
    "many shoppers walking toward the camera in a crowded shopping mall",
    "a crowd of travelers walking toward the camera at an airport arrival hall",
    "a crowd of people walking toward the camera across a busy crosswalk",
    "a crowd of students walking toward the camera on a university campus path",
    "a crowd of people walking toward the camera in a train station concourse",
    "a crowd of people walking toward the camera at a busy outdoor market street",
]
DAY_STYLE = ("cctv security camera footage, surveillance camera mounted above, slightly elevated view, "
             "upper bodies and faces visible, deep depth of field, everything in focus, realistic photo")
NIGHT_STYLE = ("night time cctv security camera footage, surveillance camera mounted above, slightly elevated "
               "view, artificial street lighting, upper bodies and faces visible, deep depth of field, "
               "everything in focus, realistic photo")
NEGATIVE = ("bokeh, depth of field, blurry, motion blur, out of focus, tilt shift, cinematic, film grain, "
            "painting, cartoon, deformed")
NIGHT_P = 0.25
KP_CONF = 0.5
MIN_EYE_DIST = 7.0      # 1152 폭 기준 — 얼굴 한 변 ~18px 이상
MIN_SLOTS = 3           # 정면 얼굴이 이보다 적으면 버린다 (뭉개진 군중 장면도 MTCNN 이 못 찾아 걸러짐)
MAX_SLOTS = 12
FACE_PER_EYE = 2.5
GENDER_LABELS = ["a photo of a man", "a photo of a woman"]


def get_window(img, x0, y0, n):
    H_, W_ = img.shape[:2]
    ax0, ay0, ax1, ay1 = max(x0, 0), max(y0, 0), min(x0 + n, W_), min(y0 + n, H_)
    return cv2.copyMakeBorder(img[ay0:ay1, ax0:ax1], ay0 - y0, (y0 + n) - ay1, ax0 - x0, (x0 + n) - ax1,
                              cv2.BORDER_REFLECT101)


def frontal_slots(img, kps, mtcnn, clip) -> list[dict]:
    slots = []
    for k in kps:
        nose, le, re = k[0], k[1], k[2]
        if min(nose[2], le[2], re[2]) < KP_CONF:
            continue
        ed = float(np.hypot(*(le[:2] - re[:2])))
        lo, hi = sorted((le[0], re[0]))
        if ed < MIN_EYE_DIST or not (lo <= nose[0] <= hi):
            continue
        fs = FACE_PER_EYE * ed
        cx, cy = (le[0] + re[0]) / 2, (le[1] + re[1]) / 2
        win = get_window(img, int(cx - fs), int(cy - fs), int(2 * fs))
        _, p = mtcnn.detect(Image.fromarray(cv2.resize(win, (192, 192), interpolation=cv2.INTER_CUBIC)[:, :, ::-1]))
        if p is None or p[0] is None or max(p) < 0.95:
            continue
        ub = get_window(img, int(cx - 1.5 * fs), int(cy - fs), int(3 * fs))
        pm = clip.classify(Image.fromarray(cv2.resize(ub, (224, 224), interpolation=cv2.INTER_CUBIC)[:, :, ::-1]),
                           GENDER_LABELS)[GENDER_LABELS[0]]
        slots.append({"leye": [round(float(v), 2) for v in le[:2]], "reye": [round(float(v), 2) for v in re[:2]],
                      "nose": [round(float(v), 2) for v in nose[:2]], "eye_dist": round(ed, 2),
                      "gender_p_male": round(pm, 3), "gender": "M" if pm > 0.65 else "F" if pm < 0.35 else "?"})
    return sorted(slots, key=lambda s: -s["eye_dist"])[:MAX_SLOTS]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=400, help="유지할 train 장면 수")
    ap.add_argument("--n-test", type=int, default=140)
    ap.add_argument("--seed", type=int, default=20261029)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = {}
    if MANIFEST.exists():           # 이어서 생성 (장면당 ~16초라 전체 2시간+)
        for l in open(MANIFEST, encoding="utf-8"):
            r = json.loads(l)
            done[r["file"]] = r
    pipe = load_scene_pipe()
    from facenet_pytorch import MTCNN
    from ultralytics import YOLO
    pose = YOLO(str(paths.ROOT / "yolo26n-pose.pt"))
    mtcnn = MTCNN(keep_all=True, device="cuda")
    clip = genmodels.ClipScorer("cuda")

    mf = open(MANIFEST, "a", encoding="utf-8")
    t0 = time.time()
    for split, n_keep, seed in (("train", args.n_train, args.seed), ("test", args.n_test, args.seed + 1)):
        rng = np.random.default_rng(seed)
        kept = sum(r["split"] == split for r in done.values())
        stats = {"attempt": 0, "few_slots": 0, "kept": kept}
        i = 0
        while stats["kept"] < n_keep:
            # 시도 i 의 프롬프트·시드는 i 로만 정해진다 → 재시작해도 같은 장면
            night = bool(rng.random() < NIGHT_P)
            prompt = f"{SCENES[rng.integers(0, len(SCENES))]}, {NIGHT_STYLE if night else DAY_STYLE}"
            s = int(rng.integers(0, 2**31 - 1))
            name = f"{split}_{i:04d}.jpg"
            i += 1
            if (OUT_DIR / name).exists() or name in done:
                continue
            stats["attempt"] += 1
            image = pipe(prompt=prompt, negative_prompt=NEGATIVE, num_inference_steps=STEPS, guidance_scale=CFG,
                         width=W, height=H, generator=torch.Generator("cuda").manual_seed(s)).images[0]
            bgr = np.array(image)[:, :, ::-1].copy()
            r = pose.predict(image, conf=0.3, verbose=False)[0]
            kps = r.keypoints.data.cpu().numpy() if r.keypoints is not None else np.zeros((0, 17, 3))
            slots = frontal_slots(bgr, kps, mtcnn, clip)
            if len(slots) < MIN_SLOTS:
                stats["few_slots"] += 1
                continue
            image.save(OUT_DIR / name, quality=95)
            rec = {"file": name, "split": split, "prompt": prompt, "seed": s, "night": night,
                   "n_persons": int(len(kps)), "slots": slots}
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            mf.flush()
            stats["kept"] += 1
            if stats["attempt"] % 20 == 0:
                print(f"  [{split}] {stats}  {time.time() - t0:.0f}s", flush=True)
        print(f"[{split}] {stats}", flush=True)
    mf.close()
    del pipe
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
