"""8차 — 저화질 얼굴인식(신원 특정): baseline vs 파인튜닝 3종 비교.

사전학습 모델  facenet-pytorch InceptionResnetV1, **CASIA-WebFace** 가중치.
               (VGGFace2 가중치는 우리 학습 신원과 같은 데이터로 학습돼 누수라 쓰지 않는다.
               CASIA-WebFace는 LFW 신원을 제거하고 만든 데이터셋이다.)

비교군 (모두 같은 스텝·배치·옵티마이저 — 데이터만 다르다)
  baseline     파인튜닝 없음
  real_deg     train 480명 실사진 clean + Albumentations 열화(p=0.5)  — 5차/6차 방식
  synth_cctv   train 480명의 생성형 CCTV 인파 얼굴만(신원 조건부 인페인팅) — bear/7차 방식
  real+synth   실사진 clean 과 생성형 CCTV 를 배치마다 1:1

손실  CosFace(s=30, m=0.35) 헤드 480클래스, 앞쪽 블록(block35 x5 까지) 동결.

평가 (scripts/31 eval_protocol.json — 학습 신원과 완전히 분리)
  1:N  5,809명 갤러리(평가 60명 등록 템플릿 + LFW 5,749명 방해 신원)에서 rank-1/5,
       TAR@FAR=1% — probe_hq / probe_lowres_nat / probe_blur_nat / probe_hq_albudeg
  1:1  LFW 6,000쌍 10-fold 정확도 — clean / albudeg
  인파 속 특정인 찾기 (scripts/33 test 프레임 + 35 검출) — 학습에 없는 60명을 생성형 CCTV
       인파 장면에서 등록 사진 5장만으로 찾기. 오인율 1% 임계값에서의 발견율.
       ※ synth_cctv 학습군과 같은 생성 파이프라인이라 그 군에 유리한 세트다 — 5차 교훈대로
         자연 저화질(probe_lowres_nat / probe_blur_nat)을 함께 봐야 한다.

출력
  runs/face/<arm>_s<seed>.pt
  reports/eval_face.json / eval_face.md
  reports/figures/fig20_face_results.png
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rareobj import albu_degrade, paths  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

ROOT = paths.DATASET / "faces_v1"
RUNS = paths.ROOT / "runs" / "face"
N_CLS = 480
STEPS, BATCH = 3000, 128
FREEZE_UNTIL = "repeat_1"   # conv stem ~ block35 x5 동결, 그 뒤만 학습
ARMS = ["baseline", "real_deg", "synth_cctv", "real+synth"]
CROWD_DETS = ROOT / "cctv_frames" / "test" / "dets.json"


def to_tensor(bgr):
    rgb = bgr[:, :, ::-1].astype(np.float32)
    return torch.from_numpy((rgb - 127.5) / 128.0).permute(2, 0, 1)   # facenet fixed standardization


class TrainSet(Dataset):
    def __init__(self, items, albu_p):
        self.items, self.albu_p = items, albu_p
        self.pipe = albu_degrade.build_pipeline() if albu_p > 0 else None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, cls = self.items[i]
        img = cv2.imread(str(ROOT / path))
        if random.random() < 0.5:
            img = img[:, ::-1]
        # 작은 위치/스케일 흔들림 — 정렬 오차 모사
        s = random.uniform(0.9, 1.0)
        c = int(160 * s)
        x, y = random.randint(0, 160 - c), random.randint(0, 160 - c)
        img = cv2.resize(np.ascontiguousarray(img[y:y + c, x:x + c]), (160, 160))
        if self.pipe is not None and random.random() < self.albu_p:
            img = albu_degrade.degrade_bgr(img, self.pipe)
        return to_tensor(img), cls


class PathSet(Dataset):
    def __init__(self, paths_, albu_seed=None):
        self.paths, self.albu_seed = paths_, albu_seed
        self.pipe = albu_degrade.build_pipeline() if albu_seed is not None else None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(str(ROOT / self.paths[i]))
        if self.pipe is not None:
            # 평가용 열화는 이미지별로 고정 — 어느 모델이든 같은 열화본을 본다
            random.seed(self.albu_seed + i)
            np.random.seed((self.albu_seed + i) % 2**32)
            img = albu_degrade.degrade_bgr(img, self.pipe)
        return to_tensor(img)


class CosFace(nn.Module):
    def __init__(self, d, n, s=30.0, m=0.35):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n, d) * 0.01)
        self.s, self.m = s, m

    def forward(self, emb, y):
        cos = F.linear(F.normalize(emb), F.normalize(self.W))
        return self.s * (cos - self.m * F.one_hot(y, cos.size(1)))


def load_backbone():
    from facenet_pytorch import InceptionResnetV1
    return InceptionResnetV1(pretrained="casia-webface", classify=False)


def train_items():
    real = [(r["path"], int(r["cls"])) for r in csv.DictReader(open(ROOT / "manifest.csv", encoding="utf-8"))
            if r["role"] == "train"]
    synth = [(r["path"], int(r["cls"])) for r in
             csv.DictReader(open(ROOT / "synth_cctv_manifest.csv", encoding="utf-8"))]
    return real, synth


def infinite(dl):
    while True:
        yield from dl


def train(arm, seed, real, synth):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = load_backbone().cuda()
    trainable, frozen = [], True
    for name, mod in model.named_children():
        for p in mod.parameters():
            p.requires_grad = not frozen
            (trainable.append(p) if not frozen else None)
        if name == FREEZE_UNTIL:
            frozen = False
    head = CosFace(512, N_CLS).cuda()
    opt = torch.optim.AdamW([{"params": trainable, "lr": 1e-4},
                             {"params": head.parameters(), "lr": 1e-2}], weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-4, 1e-2], total_steps=STEPS, pct_start=0.1)
    scaler = torch.amp.GradScaler()

    def mk(items, albu_p, bs):
        return infinite(DataLoader(TrainSet(items, albu_p), batch_size=bs, shuffle=True,
                                   num_workers=4, drop_last=True, persistent_workers=True))
    if arm == "real_deg":
        streams = [mk(real, 0.5, BATCH)]
    elif arm == "synth_cctv":
        streams = [mk(synth, 0.0, BATCH)]
    else:
        streams = [mk(real, 0.0, BATCH // 2), mk(synth, 0.0, BATCH // 2)]

    model.train()
    # BN 은 동결 구간까지 포함해 eval 고정 — 배치 통계가 작은 미세조정에서 흔들리지 않게
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
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
        if step % 250 == 0 or step == STEPS - 1:
            acc = (logits.argmax(1) == y).float().mean().item()
            print(f"    [{arm} s{seed}] step {step}/{STEPS} loss {loss.item():.3f} "
                  f"acc {acc:.3f}  {time.time() - t0:.0f}s", flush=True)
    model.eval()
    RUNS.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), RUNS / f"{arm.replace('+', '_')}_s{seed}.pt")
    return model


@torch.no_grad()
def embed(model, paths_, albu_seed=None):
    dl = DataLoader(PathSet(paths_, albu_seed), batch_size=256, num_workers=4)
    out = []
    for x in dl:
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(F.normalize(model(x.cuda()).float()).cpu())
    return dict(zip(paths_, torch.cat(out).numpy()))


def tar_at_far(gen, imp, far=0.01):
    thr = np.quantile(imp, 1 - far)
    return float((gen > thr).mean())


def identification(G, gal_ids, P, probe_ids):
    S = P @ G.T
    order = np.argsort(-S, axis=1)[:, :5]
    ranked = gal_ids[order]
    r1 = float((ranked[:, 0] == probe_ids).mean())
    r5 = float((ranked == probe_ids[:, None]).any(1).mean())
    gmask = gal_ids[None, :] == probe_ids[:, None]
    return {"rank1": r1, "rank5": r5, "tar@far1%": tar_at_far(S[gmask], S[~gmask])}


def lfw_accuracy(E, pairs):
    sims = np.array([float(E[p["a"]] @ E[p["b"]]) for p in pairs])
    same = np.array([p["same"] for p in pairs])
    fold = np.array([p["fold"] for p in pairs])
    accs = []
    for k in range(10):
        tr, te = fold != k, fold == k
        cands = np.unique(sims[tr])
        best = max(cands[::max(1, len(cands) // 400)],
                   key=lambda t: ((sims[tr] > t) == same[tr]).mean())
        accs.append(((sims[te] > best) == same[te]).mean())
    return float(np.mean(accs))


def gallery(model, proto, E=None):
    """등록 60명(5장 평균 템플릿) + LFW 방해 신원 5,749명 갤러리."""
    gid_of = proto["gid_of"]
    if E is None:
        E = embed(model, sorted(set(proto["gallery_enroll"]) | set(proto["gallery_distractors"])))
    tmpl = defaultdict(list)
    for p in proto["gallery_enroll"]:
        tmpl[gid_of[p]].append(E[p])
    gal_ids = np.array(list(tmpl) + [gid_of[p] for p in proto["gallery_distractors"]])
    G = np.stack([F.normalize(torch.tensor(np.mean(v, 0)), dim=0).numpy() for v in tmpl.values()]
                 + [E[p] for p in proto["gallery_distractors"]])
    return tmpl, G, gal_ids


def evaluate(model, proto):
    gid_of = proto["gid_of"]
    probes = dict(proto["probes"])
    need = set(proto["gallery_enroll"]) | set(proto["gallery_distractors"]) \
        | {p for v in probes.values() for p in v}
    E = embed(model, sorted(need))
    E_deg = embed(model, probes["probe_hq"], albu_seed=777)
    tmpl, G, gal_ids = gallery(model, proto, E)

    res = {}
    for name, ps in probes.items():
        res[name] = identification(G, gal_ids, np.stack([E[p] for p in ps]),
                                   np.array([gid_of[p] for p in ps]))
    res["probe_hq_albudeg"] = identification(
        G, gal_ids, np.stack([E_deg[p] for p in probes["probe_hq"]]),
        np.array([gid_of[p] for p in probes["probe_hq"]]))

    lfw_paths = sorted({p for pr in proto["lfw_pairs"] for p in (pr["a"], pr["b"])})
    res["lfw_clean"] = {"acc": lfw_accuracy(embed(model, lfw_paths), proto["lfw_pairs"])}
    res["lfw_albudeg"] = {"acc": lfw_accuracy(embed(model, lfw_paths, albu_seed=999), proto["lfw_pairs"])}
    if CROWD_DETS.exists():
        res["crowd_find"] = crowd_find(model, tmpl, G, gal_ids)
    return res


def crowd_find(model, tmpl, G, gal_ids, far=0.01):
    """인파 속 특정인 찾기 (scripts/33 test 프레임, 35 검출 결과).

    대상 인물마다 등록 사진 5장 평균 템플릿으로 프레임 속 모든 검출 얼굴과 비교한다.
    찾음 = 그 프레임에서 가장 비슷한 얼굴이 실제 그 사람이고 점수가 임계값 이상.
    임계값은 "다른 사람 얼굴을 그 사람으로 착각"하는 비율(검출 x 템플릿 비매칭 쌍)이
    1%가 되는 점 — 같은 오경보 수준에서 모델끼리 비교한다. 검출 단계에서 놓친 정답
    얼굴도 실패로 센다(end-to-end)."""
    info = json.loads(CROWD_DETS.read_text(encoding="utf-8"))
    dets = info["dets"]
    E = embed(model, [d["crop"] for d in dets])
    X = np.stack([E[d["crop"]] for d in dets])
    tg = list(tmpl)
    T = np.stack([F.normalize(torch.tensor(np.mean(tmpl[g], 0)), dim=0).numpy() for g in tg])
    S = X @ T.T                                     # (n_dets, 60)
    det_gid = np.array([d["gid"] for d in dets])
    mated = det_gid[:, None] == np.array(tg)[None, :]
    thr = float(np.quantile(S[~mated], 1 - far))
    frame_of = np.array([d["frame"] for d in dets])
    found = top1 = n_targets_det = 0
    for fi in np.unique(frame_of):
        idx = np.where(frame_of == fi)[0]
        for gid in set(det_gid[idx]) - {-1}:
            j = tg.index(gid)
            best = idx[np.argmax(S[idx, j])]
            n_targets_det += 1
            if det_gid[best] == gid:
                top1 += 1
                found += int(S[best, j] >= thr)
    # 1:N — 검출된 대상 얼굴을 5,809명 갤러리 전체에서 찾기
    tmask = det_gid >= 0
    r1n = identification(G, gal_ids, X[tmask], det_gid[tmask])
    n_gt = info["n_gt_targets"]
    return {"find_rate@far1%": found / n_gt, "find_rate_detected@far1%": found / max(n_targets_det, 1),
            "frame_top1": top1 / max(n_targets_det, 1), "threshold@far1%": thr,
            "rank1_1toN": r1n["rank1"], "detection_recall": info["detection_recall"],
            "n_gt_targets": n_gt, "n_dets": len(dets)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--arms", nargs="+", default=ARMS)
    args = ap.parse_args()

    proto = json.loads((ROOT / "eval_protocol.json").read_text(encoding="utf-8"))
    real, synth = train_items()
    print(f"train real {len(real)}  synth {len(synth)}  steps {STEPS} x batch {BATCH}")
    out_path = paths.REPORTS / "eval_face.json"
    results = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    for arm in args.arms:
        for seed in (args.seeds if arm != "baseline" else [0]):
            key = f"{arm}/s{seed}"
            ckpt = RUNS / f"{arm.replace('+', '_')}_s{seed}.pt"
            if key in results:
                if "crowd_find" in results[key] or not CROWD_DETS.exists():
                    print(f"[skip] {key}")
                    continue
                # 인파 평가셋이 나중에 생긴 경우 — 저장된 가중치로 그 평가만 추가
                print(f"[crowd-only] {key}", flush=True)
                model = load_backbone().cuda().eval()
                if arm != "baseline":
                    model.load_state_dict(torch.load(ckpt))
                tm, G, gid_ = gallery(model, proto)
                results[key]["crowd_find"] = crowd_find(model, tm, G, gid_)
            else:
                print(f"[run] {key}", flush=True)
                model = load_backbone().cuda().eval() if arm == "baseline" else train(arm, seed, real, synth)
                results[key] = evaluate(model, proto)
            print(json.dumps(results[key]), flush=True)
            out_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
            del model
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
