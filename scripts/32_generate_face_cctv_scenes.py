"""8차 — 인파 CCTV 장면 생성 (SDXL-Turbo) + 정면 인물 슬롯 검출 (YOLO26-pose).

목표 과제: "찾을 사람의 사진을 주면 CCTV 인파 속에서 그 사람을 찾아내는 AI".
그러려면 학습/평가 장면이 실제로 사람이 붐비는 공간이어야 한다 — 출퇴근 지하철
출구, 번화가 보도, 쇼핑몰·공항 등 다수 인물이 카메라 쪽으로 걸어오는 중거리 장면.

얼굴 슬롯은 YOLO26-pose 키포인트(코·양눈)로 잡는다. 처음엔 MTCNN으로 생성 인물의
얼굴을 찾으려 했지만 군중 속 얼굴은 768x512 캔버스에서 3~15px라 거의 검출되지
않았고, 신체 비율로 머리 위치만 추정하면 뒷모습 인물에도 정면 얼굴을 붙이게 돼
부자연스러웠다(사용자 피드백). pose 키포인트는 작은 인물에서도 안정적이고,
"코+양눈이 다 보임 + 코가 두 눈 사이"로 카메라를 향한 인물만 고를 수 있다. 그 눈·코
좌표가 scripts/33에서 그 인물을 다시 그릴 위치·크기 기준이 된다.

장면 split: train / test 는 서로 다른 시드로 생성한 별개의 장면이다.

출력
  data/interim/generated/face_cctv_scenes/{split}_{idx:04d}.jpg
  data/interim/generated/face_cctv_scenes_manifest.jsonl
  data/interim/generated/face_cctv_prompt_embeds.pt
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rareobj import genmodels, paths  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

GEN = paths.INTERIM / "generated"
OUT_DIR = GEN / "face_cctv_scenes"
MANIFEST = GEN / "face_cctv_scenes_manifest.jsonl"
EMBEDS = GEN / "face_cctv_prompt_embeds.pt"
W, H = 768, 512

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
# "medium distance"로는 정면 인물 눈 사이 거리 중앙값이 4.6px(얼굴 ~11px)라 인물을 그려 넣을
# 창을 10배 가까이 확대해야 했고, 그린 사람만 선명해 튀었다. "close medium shot, upper
# bodies"로 바꾸자 중앙값 7.6px(6px 이상 90%). 1024x640 해상도는 효과가 없어 768x512 유지.
DAY_STYLE = ("close medium shot, upper bodies and faces clearly visible, cctv security camera "
             "footage, slightly elevated view, realistic photo, sharp focus")
NIGHT_STYLE = ("close medium shot, upper bodies and faces clearly visible, night time cctv "
               "security camera footage, artificial street lighting, realistic photo, sharp focus")
NIGHT_P = 0.25
KP_CONF = 0.5
MIN_EYE_DIST = 6.0     # 원본 캔버스 px — x2 렌더링 시 얼굴 ~30px 이상
MAX_PER_FRAME = 10


def encode_prompts() -> dict:
    """텍스트 인코더만 담은 파이프라인으로 모든 프롬프트를 한 번에 인코딩하고 해제한다.
    UNet과 텍스트 인코더를 동시에 올리면 8GB를 넘겨 공유메모리 스왑으로 장당 13초+가
    됐다(8차에서 겪음). 분리하면 생성은 장당 0.5초, 피크 6.4GB."""
    pipe = genmodels.load_sdxl_turbo_text_encoders()
    out = {}
    with torch.no_grad():
        for p in [f"{d}, {s}" for d in SCENES for s in (DAY_STYLE, NIGHT_STYLE)]:
            pe, _, ppe, _ = pipe.encode_prompt(p, device="cuda", do_classifier_free_guidance=False)
            out[p] = (pe.cpu(), ppe.cpu())
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return out


def frontal_slots(kps: np.ndarray) -> list[dict]:
    slots = []
    for k in kps:
        nose, le, re = k[0], k[1], k[2]
        if min(nose[2], le[2], re[2]) < KP_CONF:
            continue
        ed = float(np.hypot(*(le[:2] - re[:2])))
        lo, hi = sorted((le[0], re[0]))
        if ed < MIN_EYE_DIST or not (lo <= nose[0] <= hi):
            continue
        slots.append({"leye": [round(float(v), 2) for v in le[:2]],
                      "reye": [round(float(v), 2) for v in re[:2]],
                      "nose": [round(float(v), 2) for v in nose[:2]],
                      "eye_dist": round(ed, 2)})
    return sorted(slots, key=lambda s: -s["eye_dist"])[:MAX_PER_FRAME]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20261001)
    args = ap.parse_args()

    paths.ensure()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[encode] prompts …", flush=True)
    embeds = encode_prompts()
    torch.save(embeds, EMBEDS)
    print("[load] SDXL-Turbo (UNet+VAE) / YOLO26-pose …", flush=True)
    pipe = genmodels.load_sdxl_turbo_unet_vae()
    from ultralytics import YOLO
    pose = YOLO("yolo26n-pose.pt")

    manifest = []
    for split, n, seed in (("train", args.n_train, args.seed), ("test", args.n_test, args.seed + 1)):
        rng = np.random.default_rng(seed)
        stats = {"attempt": 0, "no_slot": 0, "kept": 0, "slots": 0}
        for i in range(n):
            stats["attempt"] += 1
            night = bool(rng.random() < NIGHT_P)
            prompt = f"{SCENES[rng.integers(0, len(SCENES))]}, {NIGHT_STYLE if night else DAY_STYLE}"
            s = int(rng.integers(0, 2**31 - 1))
            pe, ppe = (t.to("cuda") for t in embeds[prompt])
            image = pipe(prompt_embeds=pe, pooled_prompt_embeds=ppe, num_inference_steps=2,
                         guidance_scale=0.0, width=W, height=H,
                         generator=torch.Generator(device="cuda").manual_seed(s)).images[0]
            r = pose.predict(image, conf=0.3, verbose=False)[0]
            kps = r.keypoints.data.cpu().numpy() if r.keypoints is not None else np.zeros((0, 17, 3))
            slots = frontal_slots(kps)
            # 처음엔 정면 인물 2명 이상만 받았는데 수율이 11%(300장 중 32장)라 1명 이상으로
            # 완화했다 — 뒷모습·측면 인물이 이미 군중을 이루고 있어 "인파" 조건은 유지된다
            if len(slots) < 1:
                stats["no_slot"] += 1
                continue
            name = f"{split}_{stats['kept']:04d}.jpg"
            image.save(OUT_DIR / name, quality=95)
            manifest.append({"file": name, "split": split, "prompt": prompt, "seed": s,
                             "night": night, "n_persons": int(len(kps)), "slots": slots})
            stats["kept"] += 1
            stats["slots"] += len(slots)
            if (i + 1) % 100 == 0:
                print(f"  [{split}] {i + 1}/{n}  {stats}", flush=True)
        print(f"[{split}] {stats}", flush=True)

    with open(MANIFEST, "w", encoding="utf-8") as f:
        for rec in manifest:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
