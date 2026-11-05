"""8차(10명) — 10명 타깃 + 모르는 사람이 섞인 생성형 CCTV 인파 프레임 (성별 맞춤).

인물 삽입은 scripts/33(신원 조건부 인페인팅, 블러 일치, 신원 검증)을 그대로 쓴다.
프레임 하나에 타깃(10명 중 1~2명)과 "모르는 사람"(실존 인물이지만 10명이 아님)을 함께
그려 넣고, 나머지 생성 인물도 모르는 사람으로 남는다.

v2 (사용자 요청) — 남자 사진을 넣을지 여자 사진을 넣을지 구분한다
  - 슬롯 성별: scripts/32b 가 장면 속 정면 인물 상반신을 CLIP 으로 판정(M/F/?)
  - 사람 성별: 타깃은 targets.csv(40에서 CLIP 판정), 모르는 사람은 등록 사진으로 CLIP 판정
  - 같은 성별끼리만 배정, "?" 슬롯은 쓰지 않는다
  - 인페인팅 프롬프트에 "a man" / "a woman" 을 명시
  - 열화는 하지 않고 깨끗한 프레임만 저장 → scripts/41b 가 CCTV 화면 생성형 스타일화 +
    렌즈·OSD·열화 + 신원 재검증 후 최종 프레임/라벨을 만든다

  train  32 train 장면 중 N_TRAIN_SCENES 개 / 타깃 사진은 faces10 train 분할에서만,
         모르는 사람은 VGGFace2 나머지 470명(unknown_train)
  test   32 test 장면 전부 / 타깃 사진은 faces10 test_hq 분할(학습에 안 쓴 사진),
         모르는 사람은 eval_vgg 60명(학습에 한 번도 안 나온 사람)

출력  data/dataset/faces10/cctv_frames/{train,test}/clean/*.jpg + frames_clean.jsonl
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import random
import sys
import time
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

c33 = importlib.import_module("33_compose_face_cctv_synth")
F10 = paths.DATASET / "faces10"
V1 = paths.DATASET / "faces_v1"
UNKNOWN = 10
N_TRAIN_SCENES = 600
MAX_TARGETS = {"train": 2, "test": 2}
N_UNKNOWN = 1
GENDER_WORD = {"M": "man", "F": "woman"}
PROMPT = {False: "photo of a {g} walking in a crowd, face looking at the camera, cctv security camera "
                 "footage, realistic, natural lighting",
          True: "photo of a {g} walking in a crowd at night, face looking at the camera, night time cctv "
                "security camera footage, artificial street lighting, realistic"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-scenes", type=int, default=0)
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--seed", type=int, default=20261001)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    split = json.loads((F10 / "split.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader(open(F10 / "manifest.csv", encoding="utf-8")))
    tgender = {int(r["cls"]): r["gender"] for r in csv.DictReader(open(F10 / "targets.csv", encoding="utf-8"))}
    v1 = {r["path"]: r for r in csv.DictReader(open(V1 / "manifest.csv", encoding="utf-8"))}
    tgt = {s: defaultdict(list) for s in ("train", "test")}
    for r in rows:
        if r["split"] == "train":
            tgt["train"][int(r["cls"])].append({"path": str(F10 / r["path"])})
        elif r["split"] == "test_hq":
            tgt["test"][int(r["cls"])].append({"path": str(F10 / r["path"])})
    unk = {s: defaultdict(list) for s in ("train", "test")}
    for p in split["unknown_train"]:
        unk["train"][v1[p]["gid"]].append({"path": str(V1 / p)})
    for p in split["unknown_test_hq"]:
        unk["test"][v1[p]["gid"]].append({"path": str(V1 / p)})

    # 모르는 사람 성별 (사진 3장 CLIP 다수결) — 인페인팅 모델 올리기 전에 끝내고 내린다
    clip = genmodels.ClipScorer("cuda")
    labels = ["a photo of a man's face", "a photo of a woman's face"]
    ugender = {}
    for sp in unk:
        for g, ps in unk[sp].items():
            v = [clip.classify(Image.open(p["path"]).convert("RGB"), labels)[labels[0]] for p in ps[:3]]
            ugender[(sp, g)] = "M" if np.mean(v) >= 0.5 else "F"
    del clip
    print("unknown gender:", {sp: dict(zip(*np.unique([v for (s, _), v in ugender.items() if s == sp],
                                                      return_counts=True))) for sp in unk})

    models = c33.Models()
    banks = {(k, s): c33.IdentityBank(models, pool[s], rng) for k, pool in (("t", tgt), ("u", unk)) for s in pool}
    scenes = [json.loads(l) for l in open(c33.SCENE_MANIFEST, encoding="utf-8")]
    stats = defaultdict(int)
    stats["cos_all"], stats["blur_sigma"] = [], []
    t0 = time.time()
    for sp in args.splits:
        sc = [s for s in scenes if s["split"] == sp]
        if sp == "train":
            sc = random.Random(1).sample(sc, min(N_TRAIN_SCENES, len(sc)))
        if args.max_scenes:
            sc = sc[:args.max_scenes]
        d = F10 / "cctv_frames" / sp
        (d / "clean").mkdir(parents=True, exist_ok=True)
        fj = open(d / "frames_clean.jsonl", "w", encoding="utf-8")
        use_t = defaultdict(int)
        unk_by_g = defaultdict(list)
        for g in unk[sp]:
            unk_by_g[ugender[(sp, g)]].append(g)
        for si, scene in enumerate(sc):
            slots = [s for s in scene["slots"] if s.get("gender") in ("M", "F")]
            if not slots:
                stats["no_gendered_slot"] += 1
                continue
            rng.shuffle(slots)
            base = cv2.imread(str(c33.SCENES / scene["file"]))
            canvas = cv2.resize(models.upscale4(base), (int(base.shape[1] * c33.F_RENDER),
                                                        int(base.shape[0] * c33.F_RENDER)), interpolation=cv2.INTER_AREA)
            # 배정: 슬롯 성별과 같은 성별의 타깃(사용 횟수 적은 순) → 남는 슬롯 하나에 같은 성별 모르는 사람
            plan, used_t = [], set()
            for s in slots:
                if sum(k == "t" for k, _, _ in plan) >= MAX_TARGETS[sp]:
                    break
                cands = [c for c in range(10) if tgender[c] == s["gender"] and c not in used_t]
                if cands:
                    c = min(cands, key=lambda c: (use_t[c], rng.random()))
                    used_t.add(c)
                    plan.append(("t", c, s))
            rest = [s for s in slots if all(s is not p[2] for p in plan)]
            for s in rest[:N_UNKNOWN]:
                plan.append(("u", rng.choice(unk_by_g[s["gender"]]), s))
            faces, untouched = [], [c33.unknown_box(s) for s in scene["slots"] if all(s is not p[2] for p in plan)]
            for kind, who, slot in plan:
                ident = banks[(kind, sp)].get(who)
                prompt = PROMPT[scene["night"]].format(g=GENDER_WORD[slot["gender"]])
                res = None if ident is None else c33.insert_person(models, canvas, slot, ident, scene["night"], rng,
                                                                   stats, prompt=prompt)
                if res is None:
                    untouched.append(c33.unknown_box(slot))
                    continue
                box, cos = res
                if kind == "t":
                    use_t[who] += 1
                faces.append({"cls": who if kind == "t" else UNKNOWN, "who": str(who), "gender": slot["gender"],
                              "box": [round(v, 1) for v in box], "id_cos": round(cos, 3)})
            if not any(f["cls"] != UNKNOWN for f in faces):
                stats["frame_no_target"] += 1
                continue
            name = Path(scene["file"]).stem
            cv2.imwrite(str(d / "clean" / f"{name}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
            fj.write(json.dumps({"image": f"clean/{name}.jpg", "scene": scene["file"], "size": [canvas.shape[1], canvas.shape[0]],
                                 "night": scene["night"], "faces": faces, "untouched_frontal": untouched},
                                ensure_ascii=False) + "\n")
            fj.flush()
            stats[f"{sp}_frames"] += 1
            stats[f"{sp}_targets"] += sum(f["cls"] != UNKNOWN for f in faces)
            stats[f"{sp}_unknowns"] += sum(f["cls"] == UNKNOWN for f in faces)
            if si % 20 == 0:
                c = stats["cos_all"]
                summ = {k: v for k, v in stats.items() if not isinstance(v, list)}
                print(f"  [{sp}] scene {si}/{len(sc)}  {summ}  id_cos med "
                      f"{np.median(c) if c else float('nan'):.3f}  {time.time() - t0:.0f}s", flush=True)
        fj.close()
        print(f"  [{sp}] target count per class: {dict(sorted(use_t.items()))}")

    c = np.array(stats["cos_all"])
    print(f"[done] {({k: v for k, v in stats.items() if not isinstance(v, list)})}  {time.time() - t0:.0f}s")
    if len(c):
        print(f"  id_cos q10 {np.quantile(c, .1):.3f}  med {np.median(c):.3f}  통과율 {(c >= c33.ID_COS_MIN).mean():.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
