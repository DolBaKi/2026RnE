"""9차 — 1단계: 같은 사진 5장에서 학습 변형을 10배로 (fewshot5e).

66 은 변형 한 장마다 얼굴 20개를 새로 스왑했다(장당 ~20s → 200 변형 × 5장이면 ~6h).
여기서는 "자리 × 타깃" 스왑을 한 번씩만 만들어 패치로 저장해 두고(스왑 은행), 변형은 패치를 붙여 조합만 바꾼다.
  타깃 은행     품질 통과한 큰 자리(≥20px) × 타깃 10명 × K_TGT 개 (원본 사진 조합을 바꿔 가며, 66 과 같은 best-of-4·2회)
  모르는 사람   남성 자리마다 K_UNK 명 (자리 나이 ± 8세)
  변형          사진마다 N_VAR 장 — 66 과 같은 나이순 배정(매번 후보·잡음 새로), 패치 붙이기
val/test 는 fewshot5d 것을 그대로 복사한다(det_mc2 와 같은 시험).

출력  data/dataset/CCTV_work/fewshot5e/{images,labels}/{train,val,test}/, meta.jsonl
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from importlib import import_module
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
m66 = import_module("66_fewshot5_diverse")
m63, CCTV, WIDER = m66.m63, m66.CCTV, m66.WIDER

SRC, OUT = m63.WORK / "fewshot5d", m63.WORK / "fewshot5e"
N_VAR, K_TGT, K_UNK = 200, 2, 4
UNKNOWN = m66.UNKNOWN


def patch(img, out):
    """스왑이 바꾼 화소만 (y0, x0, 조각, 마스크) 로."""
    d = np.any(out != img, axis=2)
    ys, xs = np.nonzero(d)
    if not len(ys):
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return int(y0), int(x0), out[y0:y1, x0:x1].copy(), d[y0:y1, x0:x1]


def paste(im, p):
    y0, x0, a, m = p
    im[y0:y0 + a.shape[0], x0:x0 + a.shape[1]][m] = a[m]


def main() -> int:
    rng = np.random.default_rng(0)
    det, sw, gate = m66.SCRFD(), m66.InSwapper(), m66.GenderGate()
    sw.detail_match = True
    ids, emb = m63.build_identities(det, sw, rng)
    T = ids["targets"]
    enr = m63.embed_images(det, sw, [p for t in T for p in t["enroll"]])
    for t in T:
        t["held"] = m63.nrm(np.mean([enr[p] for p in t["enroll"] if p in enr], 0))
        t["src_train"] = m63.select_sources(emb, t["swap_train"])
        t["age"] = round(m66.id_age(det, gate, t["src_train"][0]), 1)
    t_age = np.array([t["age"] for t in T])
    pool = ids["unknown_train"]
    u_age = np.array([m66.id_age(det, gate, u["imgs"]) for u in pool])

    for sp in ("train", "val", "test"):
        (OUT / "images" / sp).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / sp).mkdir(parents=True, exist_ok=True)
    meta = open(OUT / "meta.jsonl", "w", encoding="utf-8")
    m66.OUT = OUT                                           # m66.save 가 쓰는 출력 위치
    st, t0, cos = {}, time.time(), []
    recs = {json.loads(l)["file"]: json.loads(l) for l in open(CCTV / "pseudo_labels.jsonl", encoding="utf-8")}
    gt_tr = m66.m65.read_gt("train")

    for src, f in m66.PHOTOS:
        if src == "cctv":
            img = cv2.imread(str(CCTV / "RawDataFrames" / f))
            rec = m66.cctv_rec(det, gate, img, recs[f])
        else:
            img = cv2.imread(str(WIDER / "WIDER_train" / "images" / f))
            rec = m66.wider_rec(det, gate, img, f, gt_tr[f])
        rec = m66.mark_bad_slots(sw, img, rec, T, emb, rng)
        big = m66.slots_of(rec, m63.TARGET_MIN_W)
        # 은행: bank[(자리, cls)] = [(패치, 이름, id_cos)]
        bank = {}
        for i in big:
            kps = np.float32(rec["kps"][i])
            for t in T:
                for _ in range(K_TGT):
                    o = m63.swap_best(sw, img, kps, emb, t["src_train"], rng)
                    p = patch(img, o)
                    if p:
                        bank.setdefault((i, t["cls"]), []).append(
                            (p, t["name"], round(float(sw.embed(o, kps) @ t["held"]), 3)))
        for i in m66.slots_of(rec):
            kps = np.float32(rec["kps"][i])
            d = np.nan_to_num(np.abs(u_age - rec["age"][i]), nan=1e3)
            near = np.flatnonzero(d <= m66.UNK_AGE_TOL)
            if len(near) < 5:
                near = np.argsort(d)[:15]
            for j in rng.choice(near, min(K_UNK, len(near)), replace=False):
                u = pool[int(j)]
                ps = [p for p in u["imgs"] if p in emb]
                c = m63.nrm(np.mean([emb[p] for p in ps], 0))
                o = m63.swap_best(sw, img, kps, emb, (ps, c), rng)
                p = patch(img, o)
                if p:
                    bank.setdefault((i, UNKNOWN), []).append((p, u["name"], round(float(sw.embed(o, kps) @ c), 3)))
        print(f"  {f}  은행 {sum(len(v) for v in bank.values())}  {time.time() - t0:.0f}s", flush=True)

        m66.save("train", m66.stem(rec) + "_orig", img, rec, [], meta, st)
        for v in range(N_VAR):
            tgt, rest = m66.assign_by_age(rec, rng, t_age)
            out, faces = img.copy(), []
            jobs = [(tgt[c], t["cls"]) for c, t in enumerate(T)] + [(i, UNKNOWN) for i in rest[:m63.UNK_PER_VAR]]
            for i, c in jobs:
                cand = bank.get((i, c))
                if not cand:
                    continue
                p, name, ic = cand[int(rng.integers(len(cand)))]
                paste(out, p)
                faces.append({"face": int(i), "box": rec["boxes"][i], "kps": rec["kps"][i], "cls": c, "who": name,
                              "id_cos": ic, "slot_age": rec["age"][i]})
            cos += [x["id_cos"] for x in faces if x["cls"] < UNKNOWN]
            m66.save("train", f"{m66.stem(rec)}_v{v:03d}", out, rec, faces, meta, st)
        print(f"  {f}  변형 {N_VAR}  {time.time() - t0:.0f}s", flush=True)

    # val/test 는 fewshot5d 그대로
    for l in open(SRC / "meta.jsonl", encoding="utf-8"):
        m = json.loads(l)
        if m["split"] == "train":
            continue
        for a in (m["image"], m["image"].replace("images/", "labels/").replace(".jpg", ".txt")):
            shutil.copy(SRC / a, OUT / a)
        meta.write(l if l.endswith("\n") else l + "\n")
        st[m["split"]] = st.get(m["split"], 0) + 1
    meta.close()
    print("[done]", st, f"train cos 중앙 {np.median(cos):.3f} n{len(cos)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
