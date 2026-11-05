"""9차 장면 다양화 — 사진 5장(모스크 CCTV 1 + WIDER 4) × 타깃 10명, 5장 모두에 10명이 다 들어간다.

사용자 결정(2026-11-02): 모스크 CCTV 만으로는 장면이 한 종류라 "모스크 1 + WIDER 4" 로 바꾼다.
타깃은 63 과 같은 남성 10명(identities.json 재사용), 스왑은 CLIP 이 남성으로 본 자리에만 넣는다.

  사진 5장   모스크 cam06(실내·부감) / WIDER 밤 거리 · 회의실 · 밀집 시위(부감) · 행진(야외)
             (처음 고른 쇼핑 거리·시위 사진은 남성 20px 자리가 6·9개뿐이라 타깃 10명이 다 안 들어가 바꿨고,
              회의실 301 은 남성 판정을 통과한 자리 중 여성이 섞여 있어(눈 확인) 566 으로 바꿨다)
  남성 판정  CLIP 상반신 · CLIP 얼굴 · InsightFace genderage 셋 다 ≥ MALE_P 일 때만 남성 자리.
             CLIP 상반신 하나로는 20px 안팎에서 여성(귀걸이·긴 머리)을 남성으로 본 경우가 있었다.
  라벨       WIDER = 사람이 그린 박스(무효 표시 제외), 모스크 = SCRFD 의사 라벨 + 다중 배율 보강(63)
  자리       WIDER 박스와 IoU 로 맞는 SCRFD 검출(점수≥0.6, 정면)의 5점, 가림·자세 0, 남성(CLIP)
  스왑       63 과 같음 — 선명한 소스, 후보 4개 × 2회 스왑, 타깃은 폭 ≥ 20px 자리
             + 고주파 세기를 원래 얼굴에 맞춤(큰 얼굴이 뭉개져 붙여 넣은 티), 시험 스왑 cos 가 낮은 자리 제외
  나이       genderage 나이로 타깃을 자리에 나이순 배정(헝가리안), 모르는 사람은 자리 나이 ±8세에서 고른다
             (사용자 지적 2026-11-02 — 타깃 유지 결정, 타깃 36~64세 · 군중 대부분 22~45세라 근사만 된다)
  val/test   WIDER_val 군중 범주(사람 라벨, 학습에 안 쓴 사진) + 모스크 cam06 을 뺀 카메라.
             val = 모스크 cam19·22 + WIDER_val 군중의 1/5, test = 나머지. 결과는 출처별로 따로 본다.

출력  data/dataset/CCTV_work/fewshot5d/{images,labels}/{train,val,test}/, meta.jsonl, data.yaml, targets.json
"""

from __future__ import annotations

import json
import sys
import time
import zlib
from importlib import import_module
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rareobj import genmodels, paths  # noqa: E402
from rareobj.cctv_faceswap import SCRFD, InSwapper  # noqa: E402

m63 = import_module("63_cctv_fewshot5")
m65 = import_module("65_wider_candidates")

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

CCTV, WIDER = m63.CCTV, m65.WIDER
OUT = m63.WORK / "fewshot5d"
PHOTOS = [("cctv", "IMG-43926.png"),
          ("wider", "2--Demonstration/2_Demonstration_Protesters_2_226.jpg"),             # 밤 거리(동아시아)
          ("wider", "11--Meeting/11_Meeting_Meeting_11_Meeting_Meeting_11_566.jpg"),      # 실내 모임
          ("wider", "2--Demonstration/2_Demonstration_Demonstration_Or_Protest_2_17.jpg"),  # 밀집 시위 부감(유럽)
          ("wider", "10--People_Marching/10_People_Marching_People_Marching_2_338.jpg")]
GENDER = ["a photo of a man", "a photo of a woman"]
MALE_P = 0.65                       # 이보다 확신 낮으면 남성 자리로 쓰지 않는다(32b MARGIN 0.15 와 같음)
WIDER_EVAL_MAX, WIDER_VAL_FRAC = 200, 0.2
CAND_FRAC, AGE_NOISE = 0.6, 5.0     # 나이순 배정: 후보 자리 비율, 나이 비용 잡음(세)
UNK_AGE_TOL = 8.0                   # 모르는 사람은 자리 나이 ± 이 범위 안에서 고른다
Q_MIN = 0.45                        # 학습 사진 자리 품질(시험 스왑 신원 cos) 하한
N_VAR, N_TARGET, UNK_PER_VAR = m63.N_VAR, m63.N_TARGET, m63.UNK_PER_VAR
EVAL_TGT, EVAL_UNK, UNKNOWN = m63.EVAL_TGT, m63.EVAL_UNK, m63.UNKNOWN


FACE_GENDER = ["a photo of a man's face", "a photo of a woman's face"]


class GenderGate:
    """남성 확률 = min(CLIP 상반신, CLIP 얼굴, InsightFace genderage) — 셋이 모두 남성이라 해야 높다."""

    def __init__(self, device="cuda"):
        from onnx2torch import convert
        from rareobj.cctv_faceswap import _load_onnx

        self.clip = genmodels.ClipScorer(device)
        self.ga = convert(_load_onnx("public-data/insightface", "models/buffalo_l/genderage.onnx")).to(device).eval()
        self.device = device

    def body(self, img, kps) -> float:
        """상반신 크롭(머리 위 0.6 ~ 아래 4 얼굴, 폭 3 얼굴) — scripts/32b 와 같은 크롭."""
        k = np.float32(kps)
        fs = 2.5 * np.linalg.norm(k[1] - k[0])
        cx, cy = (k[0] + k[1]) / 2
        H, W = img.shape[:2]
        x0, x1 = int(max(cx - 1.5 * fs, 0)), int(min(cx + 1.5 * fs, W))
        y0, y1 = int(max(cy - 0.6 * fs, 0)), int(min(cy + 4.0 * fs, H))
        crop = cv2.resize(img[y0:y1, x0:x1], None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        return self.clip.classify(Image.fromarray(crop[:, :, ::-1]), GENDER)[GENDER[0]]

    def face(self, img, box) -> float:
        x0, y0, x1, y1 = box
        m = 0.35 * (x1 - x0)
        c = img[int(max(y0 - m, 0)):int(y1 + m), int(max(x0 - m, 0)):int(x1 + m)]
        c = cv2.resize(c, (224, 224), interpolation=cv2.INTER_CUBIC)
        return self.clip.classify(Image.fromarray(c[:, :, ::-1]), FACE_GENDER)[FACE_GENDER[0]]

    @torch.no_grad()
    def _ga(self, img, box) -> np.ndarray:
        """insightface attribute.py 와 같은 입력: 박스 중심, 긴 변 × 1.5 를 96px, RGB 0~255.
        출력 [여성, 남성 로짓, 나이/100]."""
        x0, y0, x1, y1 = box
        s = 96 / (max(x1 - x0, y1 - y0) * 1.5)
        M = np.float32([[s, 0, 48 - (x0 + x1) / 2 * s], [0, s, 48 - (y0 + y1) / 2 * s]])
        a = cv2.warpAffine(img, M, (96, 96), flags=cv2.INTER_CUBIC)
        x = torch.from_numpy(a[:, :, ::-1].astype(np.float32)).permute(2, 0, 1)[None].to(self.device)
        return self.ga(x)[0].float().cpu().numpy()

    def genderage(self, img, box) -> float:
        y = self._ga(img, box)
        return float(torch.from_numpy(y[:2]).softmax(0)[1])  # 1 = 남성

    def age(self, img, box) -> float:
        return float(self._ga(img, box)[2] * 100)

    def __call__(self, img, box, kps) -> float:
        return min(self.body(img, kps), self.face(img, box), self.genderage(img, box))


def wider_rec(det, gate, img, f, gt) -> dict:
    """WIDER 사진 → 63 형식 rec: boxes = 사람 박스, kps = 맞춘 SCRFD 5점, slot = 스왑 가능 여부."""
    G, kps, sc, fr = m65.match_slots(det, img, gt)
    ok = (gt[:, 7] == 0) & (gt[:, 8] == 0) & (gt[:, 9] == 0)
    slot = [bool(ok[i] and kps[i] is not None and sc[i] >= m63.SLOT_THR and fr[i]
                 and G[i, 2] - G[i, 0] >= m63.SLOT_MIN_W) for i in range(len(G))]
    mp = [round(gate(img, G[i], kps[i]), 3) if slot[i] else None for i in range(len(G))]
    sl = [s and p >= MALE_P for s, p in zip(slot, mp)]
    return {"file": f, "cam": f.split("/")[0], "w": img.shape[1], "h": img.shape[0], "src": "wider",
            "boxes": G.round(2).tolist(), "label": (gt[:, 7] == 0).tolist(),
            "kps": [k if k is not None else [[0, 0]] * 5 for k in kps],
            "slot": sl, "male_p": mp, "age": [round(gate.age(img, G[i]), 1) if sl[i] else None for i in range(len(G))]}


def cctv_rec(det, gate, img, rec) -> dict:
    """모스크 프레임 → 같은 형식. 라벨 = 의사 라벨(≥0.5) + 다중 배율 보강, 자리 = 63 자리 중 남성."""
    rec = m63.refine_labels(det, img, rec)
    n = len(rec["boxes"])
    S = set(m63.slots_of(rec))
    mp = [round(gate(img, rec["boxes"][i], rec["kps"][i]), 3) if i in S else None for i in range(n)]
    extra = rec["extra_boxes"]
    sl = [i in S and mp[i] >= MALE_P for i in range(n)]
    return {"file": rec["file"], "cam": rec["cam"], "w": rec["w"], "h": rec["h"], "src": "cctv",
            "boxes": rec["boxes"] + extra,
            "label": [s >= m63.LABEL_THR for s in rec["scores"]] + [True] * len(extra),
            "kps": rec["kps"] + [[[0, 0]] * 5] * len(extra),
            "slot": sl + [False] * len(extra),
            "male_p": mp + [None] * len(extra),
            "age": [round(gate.age(img, rec["boxes"][i]), 1) if sl[i] else None for i in range(n)]
                   + [None] * len(extra)}


def slots_of(rec, min_w=m63.SLOT_MIN_W) -> list[int]:
    bad = set(rec.get("bad", ()))
    return [i for i, (b, s) in enumerate(zip(rec["boxes"], rec["slot"]))
            if s and b[2] - b[0] >= min_w and i not in bad]


def photo_age(det, gate, p) -> float | None:
    """등록·소스 사진(얼굴 하나가 크게 나온 사진)의 가장 큰 얼굴 나이."""
    im = cv2.imread(p)
    b, _, _ = det.detect(im, scale=1.0, thr=0.5)
    if not len(b):
        return None
    return gate.age(im, b[int(np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])))])


def id_age(det, gate, ps) -> float:
    a = [x for x in (photo_age(det, gate, p) for p in ps) if x is not None]
    return float(np.median(a)) if a else float("nan")


def assign_by_age(rec, rng, ages_t):
    """나이순 배정 — 큰 자리 중 무작위 일부(CAND_FRAC)를 후보로, |타깃 나이 − 자리 나이| + 잡음 합이 최소.
    타깃(36~64세)이 군중(대부분 22~45세)보다 나이가 많아 완전히 맞출 수는 없고, 나이 많은 자리로 몰아준다.
    후보를 매번 다시 뽑고 잡음을 더해 변형마다 같은 자리에만 들어가지 않게 한다. 반환: 타깃별 자리, 나머지 자리."""
    big = slots_of(rec, m63.TARGET_MIN_W)
    cand = [int(i) for i in rng.permutation(big)[:max(len(ages_t), int(round(CAND_FRAC * len(big))))]]
    sa = np.array([rec["age"][i] for i in cand])
    C = np.abs(np.asarray(ages_t)[:, None] - sa[None]) + rng.uniform(0, AGE_NOISE, (len(ages_t), len(cand)))
    r, c = linear_sum_assignment(C)
    tgt = [cand[j] for j in c[np.argsort(r)]]
    rest = [int(i) for i in rng.permutation([i for i in slots_of(rec) if i not in tgt])]
    return tgt, rest


def mark_bad_slots(sw, img, rec, T, emb, rng):
    """학습 사진의 스왑 품질 거르기 — 자리마다 타깃 2명을 한 번씩 넣어 본 신원 cos 평균이 Q_MIN 미만이면
    (손에 가림·심한 흐림 등) 그 자리는 아예 쓰지 않는다. 타깃 수 + 2 자리는 남긴다."""
    big = slots_of(rec, m63.TARGET_MIN_W)
    q = {}
    for i in big:
        kps = np.float32(rec["kps"][i])
        s = []
        for c in rng.choice(len(T), 2, replace=False):
            o, _ = sw.swap(img, kps, m63.mean_emb(emb, T[c]["src_train"][0], rng))
            s.append(float(sw.embed(o, kps) @ T[c]["held"]))
        q[i] = float(np.mean(s))
    order = sorted(big, key=lambda i: -q[i])
    rec["bad"] = [i for i in order[N_TARGET + 2:] if q[i] < Q_MIN]
    rec["slot_q"] = {int(i): round(v, 3) for i, v in q.items()}
    return rec


def yolo_lines(rec) -> list[str]:
    W, H = rec["w"], rec["h"]
    out = []
    for (x0, y0, x1, y1), keep in zip(rec["boxes"], rec["label"]):
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        if keep and x1 - x0 >= 2 and y1 - y0 >= 2:
            out.append(f"0 {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
    return out


def save(split, name, im, rec, faces, meta, st):
    cv2.imwrite(str(OUT / "images" / split / f"{name}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, m63.JPEG_Q])
    lines = yolo_lines(rec)
    (OUT / "labels" / split / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
    meta.write(json.dumps({"image": f"images/{split}/{name}.jpg", "split": split, "src": rec["src"],
                           "src_frame": rec["file"], "cam": rec["cam"], "n_faces": len(lines),
                           "n_male_slots": len(slots_of(rec)), "swaps": faces}) + "\n")
    st[f"{split}/{rec['src']}"] = st.get(f"{split}/{rec['src']}", 0) + 1


def stem(rec) -> str:
    return Path(rec["file"]).stem if rec["src"] == "cctv" else "w_" + Path(rec["file"]).stem


def main() -> int:
    rng = np.random.default_rng(0)
    det, sw, gate = SCRFD(), InSwapper(), GenderGate()
    sw.detail_match = True                                  # 큰 얼굴에서 스왑이 뭉개지는 것 보정
    ids, emb = m63.build_identities(det, sw, rng)          # fewshot5 캐시 — 같은 타깃 10명
    T = ids["targets"]
    enr = m63.embed_images(det, sw, [p for t in T for p in t["enroll"]])
    for t in T:
        t["held"] = m63.nrm(np.mean([enr[p] for p in t["enroll"] if p in enr], 0))
        t["src_train"] = m63.select_sources(emb, t["swap_train"])
        t["src_test"] = m63.select_sources(emb, t["swap_test"])
        t["age"] = round(id_age(det, gate, t["src_train"][0]), 1)
    t_age = np.array([t["age"] for t in T])
    u_age = {k: np.array([id_age(det, gate, u["imgs"]) for u in ids[k]]) for k in ("unknown_train", "unknown_test")}
    print("타깃 나이", t_age.tolist(), flush=True)

    def unk(pool, age):
        d = np.nan_to_num(np.abs(u_age[pool] - age), nan=1e3)
        near = np.flatnonzero(d <= UNK_AGE_TOL)
        if len(near) < 5:
            near = np.argsort(d)[:15]
        u = ids[pool][int(rng.choice(near))]
        ps = [p for p in u["imgs"] if p in emb]
        return u["name"], (ps, m63.nrm(np.mean([emb[p] for p in ps], 0)))

    def add_ages(rec, faces):
        for x in faces:
            x["slot_age"] = rec["age"][x["face"]]
        return faces

    for sp in ("train", "val", "test"):
        (OUT / "images" / sp).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / sp).mkdir(parents=True, exist_ok=True)
    (OUT / "previews").mkdir(exist_ok=True)
    meta = open(OUT / "meta.jsonl", "w", encoding="utf-8")
    st, t0, cos = {}, time.time(), {"train": [], "test_cctv": [], "test_wider": []}
    recs = {json.loads(l)["file"]: json.loads(l) for l in open(CCTV / "pseudo_labels.jsonl", encoding="utf-8")}
    gt_tr, gt_va = m65.read_gt("train"), m65.read_gt("val")

    # 학습: 사진 5장 × N_VAR 변형 (+ 원본 1장)
    for src, f in PHOTOS:
        if src == "cctv":
            img = cv2.imread(str(CCTV / "RawDataFrames" / f))
            rec = cctv_rec(det, gate, img, recs[f])
        else:
            img = cv2.imread(str(WIDER / "WIDER_train" / "images" / f))
            rec = wider_rec(det, gate, img, f, gt_tr[f])
        if len(slots_of(rec, m63.TARGET_MIN_W)) < N_TARGET:
            raise SystemExit(f"{f}: 남성·{m63.TARGET_MIN_W}px 이상 자리 {len(slots_of(rec, 20))}개 < 타깃 {N_TARGET}명")
        rec = mark_bad_slots(sw, img, rec, T, emb, rng)
        n_big = len(slots_of(rec, m63.TARGET_MIN_W))
        save("train", stem(rec) + "_orig", img, rec, [], meta, st)
        for v in range(N_VAR):
            tgt, rest = assign_by_age(rec, rng, t_age)
            assign = [(tgt[c], t["cls"], t["name"], t["src_train"], t["held"]) for c, t in enumerate(T)]
            for i in rest[:UNK_PER_VAR]:
                n, s = unk("unknown_train", rec["age"][i])
                assign.append((i, UNKNOWN, n, s, s[1]))
            out, faces = m63.compose(sw, img, rec, assign, emb, rng)
            cos["train"] += [x["id_cos"] for x in faces if x["cls"] < UNKNOWN]
            save("train", f"{stem(rec)}_v{v:02d}", out, rec, add_ages(rec, faces), meta, st)
        print(f"  {f}  라벨 {len(yolo_lines(rec))}  남성 자리 {len(slots_of(rec))} (≥20px {n_big},"
              f" 품질 탈락 {len(rec['bad'])})  {time.time() - t0:.0f}s", flush=True)

    def eval_frame(sp, rec, img, key):
        tg = rng.choice(len(T), EVAL_TGT, replace=False)
        tgt, rest = assign_by_age(rec, rng, t_age[tg])
        assign = [(tgt[k], T[c]["cls"], T[c]["name"], T[c]["src_test"], T[c]["held"]) for k, c in enumerate(tg)]
        for i in rest[:EVAL_UNK]:
            n, s = unk("unknown_test", rec["age"][i])
            assign.append((i, UNKNOWN, n, s, s[1]))
        out, faces = m63.compose(sw, img, rec, assign, emb, rng)
        add_ages(rec, faces)
        if sp == "test":
            cos[key] += [x["id_cos"] for x in faces if x["cls"] < UNKNOWN]
        save(sp, stem(rec), out, rec, faces, meta, st)

    # 모스크: cam06 을 뺀 카메라 (val = cam19·22)
    rs = sorted([r for r in recs.values() if r["cam"] != recs[PHOTOS[0][1]]["cam"]], key=lambda r: r["idx"])
    rs = [r for r in rs[::m63.EVAL_STRIDE] if len(m63.slots_of(r, m63.TARGET_MIN_W)) >= EVAL_TGT
          and len(m63.slots_of(r)) >= EVAL_TGT + EVAL_UNK][:m63.EVAL_MAX]
    n = 0
    for r in rs:
        img = cv2.imread(str(CCTV / "RawDataFrames" / r["file"]))
        rec = cctv_rec(det, gate, img, r)
        if len(slots_of(rec, m63.TARGET_MIN_W)) < EVAL_TGT or len(slots_of(rec)) < EVAL_TGT + EVAL_UNK:
            continue
        eval_frame("val" if r["cam"] in m63.VAL_CAMS else "test", rec, img, "test_cctv")
        n += 1
    print(f"  [모스크] {n}장  {time.time() - t0:.0f}s", flush=True)

    # WIDER_val 군중 범주 — 파일 이름 해시로 val/test 고정 분할
    fs = sorted(f for f in gt_va if f.split("/")[0] in m65.CROWD)
    rng.shuffle(fs)
    n = 0
    for f in fs:
        if n >= WIDER_EVAL_MAX:
            break
        g = gt_va[f]
        big = (g[:, 7] == 0) & (g[:, 8] == 0) & (g[:, 9] == 0) & (g[:, 2] >= m63.TARGET_MIN_W)
        if big.sum() < EVAL_TGT:
            continue
        img = cv2.imread(str(WIDER / "WIDER_val" / "images" / f))
        rec = wider_rec(det, gate, img, f, g)
        if len(slots_of(rec, m63.TARGET_MIN_W)) < EVAL_TGT or len(slots_of(rec)) < EVAL_TGT + EVAL_UNK:
            continue
        sp = "val" if zlib.crc32(f.encode()) % 100 < WIDER_VAL_FRAC * 100 else "test"
        eval_frame(sp, rec, img, "test_wider")
        n += 1
    print(f"  [WIDER_val] {n}장  {time.time() - t0:.0f}s", flush=True)

    meta.close()
    for t in T:
        t["src_train"], t["src_test"] = t["src_train"][0], t["src_test"][0]
        del t["held"]
    (OUT / "targets.json").write_text(json.dumps(T, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "data.yaml").write_text(f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\n"
                                   f"test: images/test\nnames:\n  0: face\n", encoding="utf-8")
    print("[done]", st, {k: f"med {np.median(v):.3f} n{len(v)}" for k, v in cos.items() if v}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
