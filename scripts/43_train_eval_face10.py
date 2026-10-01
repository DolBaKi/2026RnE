"""8차(10명) — "CCTV 인파 속 10명 찾기": baseline vs 파인튜닝 3종.

모델  facenet-pytorch InceptionResnetV1 (CASIA-WebFace 사전학습, VGGFace2 미사용 — 누수 방지)
학습  CosFace 헤드 11클래스 = 타깃 10명 + "모르는 사람" 1클래스(배경 클래스). 모르는
      사람 클래스가 임베딩에서 타깃들을 군중으로부터 떼어 놓도록 학습한다.
      앞쪽 블록 동결, 모든 군 같은 스텝·배치 — 데이터만 다르다.

  baseline     파인튜닝 없음
  real_deg     타깃 실사진(train 분할) + 모르는 사람 실사진, Albumentations 열화 p=0.5
  synth_cctv   scripts/41 생성형 CCTV 인파 얼굴만 (타깃 + 삽입한 모르는 사람)
  real+synth   실사진과 생성형 CCTV를 배치마다 섞음

추론(평가·find_person 공통)  각 타깃의 등록 사진 5장 평균 템플릿과 코사인 비교 →
      가장 비슷한 타깃 + 임계값. 임계값은 "모르는 사람을 10명 중 누군가로 착각"하는
      비율이 1%가 되는 점(FAR 1%) — 같은 오경보 수준에서 모델을 비교한다.

평가 (전부 학습에 안 쓴 사진/사람/장면)
  test_hq / test_lowres_nat   타깃 실사진 (고화질 / **원래부터** 저해상도)
  test_hq_albudeg             test_hq + Albumentations 열화 (자기 열화 과적합 점검용 대조군)
  unknown (eval_vgg 60명 실사진) 으로 FAR 1% 임계값을 잡는다
  crowd       scripts/41 test 프레임에서 MTCNN이 검출한 모든 얼굴 (35 --data faces10)
              → 타깃 등장 1회당 "맞는 사람으로 찾았는가"(검출 실패 포함), 임계값은
                프레임 속 모르는 사람 얼굴 기준 FAR 1%
              ※ synth_cctv 학습군과 같은 생성 파이프라인이라 그 군에 유리 — 자연 저해상도와 함께 볼 것

출력  runs/face10/<arm>_s<seed>.pt, reports/eval_face10.json
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rareobj import paths  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

t34 = importlib.import_module("34_train_eval_face")   # TrainSet/PathSet/CosFace/embed 재사용 (절대경로로 넘김)
F10 = paths.DATASET / "faces10"
V1 = paths.DATASET / "faces_v1"
RUNS = paths.ROOT / "runs" / "face10"
OUT_JSON = paths.REPORTS / "eval_face10.json"
N_T, UNKNOWN = 10, 10
STEPS = 1200
B_TGT, B_UNK = 96, 32
ARMS = ["baseline", "real_deg", "synth_cctv", "real+synth"]
FAR = 0.01


def load_items():
    split = json.loads((F10 / "split.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader(open(F10 / "manifest.csv", encoding="utf-8")))
    real_t = [(str(F10 / r["path"]), int(r["cls"])) for r in rows if r["split"] == "train"]
    real_u = [(str(V1 / p), UNKNOWN) for p in split["unknown_train"]]
    syn = [(str(F10 / r["path"]), int(r["cls"])) for r in csv.DictReader(open(F10 / "synth_manifest.csv", encoding="utf-8"))]
    syn_t = [x for x in syn if x[1] != UNKNOWN]
    syn_u = [x for x in syn if x[1] == UNKNOWN]
    return split, rows, real_t, real_u, syn_t, syn_u


def train(arm, seed, real_t, real_u, syn_t, syn_u):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = t34.load_backbone().cuda()
    trainable, frozen = [], True
    for name, mod in model.named_children():
        for p in mod.parameters():
            p.requires_grad = not frozen
            if not frozen:
                trainable.append(p)
        if name == t34.FREEZE_UNTIL:
            frozen = False
    head = t34.CosFace(512, N_T + 1).cuda()
    opt = torch.optim.AdamW([{"params": trainable, "lr": 1e-4}, {"params": head.parameters(), "lr": 1e-2}],
                            weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-4, 1e-2], total_steps=STEPS, pct_start=0.1)
    scaler = torch.amp.GradScaler()

    def mk(items, albu_p, bs):
        return t34.infinite(torch.utils.data.DataLoader(t34.TrainSet(items, albu_p), batch_size=bs, shuffle=True,
                                                        num_workers=3, drop_last=True, persistent_workers=True))
    if arm == "real_deg":
        streams = [mk(real_t, 0.5, B_TGT), mk(real_u, 0.5, B_UNK)]
    elif arm == "synth_cctv":
        streams = [mk(syn_t, 0.0, B_TGT), mk(syn_u, 0.0, B_UNK)]
    else:   # real+synth
        streams = [mk(real_t, 0.0, B_TGT // 2), mk(syn_t, 0.0, B_TGT // 2),
                   mk(real_u, 0.0, B_UNK // 2), mk(syn_u, 0.0, B_UNK // 2)]
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()
    t0 = time.time()
    for step in range(STEPS):
        xs, ys = zip(*(next(s) for s in streams))
        x, y = torch.cat(xs).cuda(non_blocking=True), torch.cat(ys).cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = head(model(x).float(), y)
            loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % 200 == 0 or step == STEPS - 1:
            acc = (logits.argmax(1) == y).float().mean().item()
            print(f"    [{arm} s{seed}] step {step}/{STEPS} loss {loss.item():.3f} acc {acc:.3f} "
                  f"{time.time() - t0:.0f}s", flush=True)
    model.eval()
    RUNS.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), RUNS / f"{arm.replace('+', '_')}_s{seed}.pt")
    return model


def scores(E, paths_, T):
    X = np.stack([E[p] for p in paths_])
    S = X @ T.T
    return S.max(1), S.argmax(1)


def evaluate(model, split, rows):
    cls_of = {str(F10 / r["path"]): int(r["cls"]) for r in rows}
    sets = {k: [str(F10 / p) for p in split[k]] for k in ("enroll", "test_hq", "test_lowres_nat")}
    unk = [str(V1 / p) for p in split["unknown_test_hq"] + split["unknown_test_lowres_nat"]]
    unk_low = [str(V1 / p) for p in split["unknown_test_lowres_nat"]]
    E = t34.embed(model, sorted(set(sets["enroll"] + sets["test_hq"] + sets["test_lowres_nat"] + unk)))
    E_deg = t34.embed(model, sets["test_hq"], albu_seed=777)
    T = np.stack([np.mean([E[p] for p in sets["enroll"] if cls_of[p] == c], 0) for c in range(N_T)])
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    u_max, _ = scores(E, unk, T)
    thr = float(np.quantile(u_max, 1 - FAR))
    ul_max, _ = scores(E, unk_low, T)
    res = {"threshold@far1%": thr, "unknown_lowres_far": float((ul_max >= thr).mean())}
    for name, ps, EE in (("test_hq", sets["test_hq"], E), ("test_lowres_nat", sets["test_lowres_nat"], E),
                         ("test_hq_albudeg", sets["test_hq"], E_deg)):
        smax, sarg = scores(EE, ps, T)
        y = np.array([cls_of[p] for p in ps])
        res[name] = {"top1": float((sarg == y).mean()), "dir@far1%": float(((sarg == y) & (smax >= thr)).mean()),
                     "n": len(ps)}
    dets_p = F10 / "cctv_frames" / "test" / "dets.json"
    if dets_p.exists():
        info = json.loads(dets_p.read_text(encoding="utf-8"))
        dets = info["dets"]
        Ed = t34.embed(model, [str(F10 / d["crop"]) for d in dets])
        X = np.stack([Ed[str(F10 / d["crop"])] for d in dets])
        S = X @ T.T
        smax, sarg = S.max(1), S.argmax(1)
        g = np.array([d["gid"] for d in dets])
        tc = float(np.quantile(smax[g < 0], 1 - FAR))
        tgt = g >= 0
        found = int(((sarg == g) & (smax >= tc) & tgt).sum())
        res["crowd"] = {"find_rate@far1%": found / info["n_gt_targets"],
                        "find_rate_detected@far1%": found / max(int(tgt.sum()), 1),
                        "top1_detected": float((sarg[tgt] == g[tgt]).mean()),
                        "threshold@far1%": tc, "detection_recall": info["detection_recall"],
                        "n_gt_targets": info["n_gt_targets"], "n_unknown_dets": int((g < 0).sum())}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--redo", action="store_true", help="저장된 가중치로 평가만 다시")
    args = ap.parse_args()
    split, rows, real_t, real_u, syn_t, syn_u = load_items()
    print(f"real targets {len(real_t)}  real unknown {len(real_u)}  synth targets {len(syn_t)}  synth unknown {len(syn_u)}")
    results = json.loads(OUT_JSON.read_text(encoding="utf-8")) if OUT_JSON.exists() else {}
    for arm in args.arms:
        for seed in (args.seeds if arm != "baseline" else [0]):
            key = f"{arm}/s{seed}"
            ckpt = RUNS / f"{arm.replace('+', '_')}_s{seed}.pt"
            if key in results and not args.redo:
                print(f"[skip] {key}")
                continue
            print(f"[run] {key}", flush=True)
            model = t34.load_backbone().cuda().eval()
            if arm != "baseline":
                if args.redo and ckpt.exists():
                    model.load_state_dict(torch.load(ckpt))
                else:
                    model = train(arm, seed, real_t, real_u, syn_t, syn_u)
            results[key] = evaluate(model, split, rows)
            print(json.dumps(results[key]), flush=True)
            OUT_JSON.write_text(json.dumps(results, indent=1), encoding="utf-8")
            del model
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
