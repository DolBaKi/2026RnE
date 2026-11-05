"""9차 — CCTV 사진 5장 × 타깃 10명: 5장 모두에 10명이 다 들어가는 퓨샷 데이터.

사용자 설계(2026-10-30): 고르게 분포된 CCTV 사진 5장을 고르고, 5장 각각의 클래스 목록 교집합에
타깃 10명이 모두 들어가야 한다. 실제 영상에는 타깃이 없으므로 inswapper 로 넣는다.

  사진 5장   서로 다른 카메라, 정면 얼굴(점수≥0.6, 폭≥14px)이 4x3 격자 전체에 퍼진 프레임 (PHOTOS)
  타깃 10명  전부 남성 — 영상이 남성 예배 구역이라 여성 타깃은 수염이 사라지는 등 티가 났다.
             faces10 남성 5명 + faces_v1 학습 신원 중 CLIP 남성에서 시드로 무작위 5명.
             신원마다 사진 150장을 enroll 5 / swap_train 70 / swap_test 25 / test_hq 50 으로 나눈다.
  변형본     사진마다 N_VAR 장: 타깃 10명을 매번 다른 얼굴 자리에, 그리고 모르는 사람도 같은 방식으로
             스왑한다 — "스왑된 얼굴 = 타깃" 지름길을 막기 위해(v3 교훈). 나머지는 진짜 군중.
  피부톤     스왑 얼굴을 원래 사람의 볼 피부 톤에 맞춘다(cctv_faceswap) → 목·손과 이어진다.
             타깃은 톤과 무관하게 아무 자리에나 들어가므로 타깃별 톤 편향이 없다.
  val/test   나머지 카메라 프레임에 타깃 3명 + 모르는 사람 3명을 넣는다. 타깃 신원은 swap_test 사진,
             모르는 사람은 학습에 안 쓴 신원 — 학습과 같은 사진·사람이 섞이지 않게.

품질 개선(2026-11-02, 284자리 비교 — 선택에 안 쓴 등록 사진 기준 신원 유사도 중앙값 0.44 → 0.53):
  소스 선별   흐린 사진(선명도 하위 절반) 빼고 그 사람다운(중심 유사도 상위) SRC_TOP 장만 쓴다
  최고 후보   사진 조합 N_TRY 개 × N_PASS 회 스왑 중 신원이 가장 잘 들어간 것 (2회 = 결과를 다시 스왑)
  타깃 자리   폭 TARGET_MIN_W px 이상 — 더 작으면 화소가 모자라 타깃처럼 안 보인다. 작은 자리는 모르는 사람
  라벨 보강   쓰는 프레임만 SCRFD 2·3배 × 좌우반전 4회 중 2회 이상 잡힌 얼굴(평균 점수≥0.5)을 더한다

출력  data/dataset/CCTV_work/fewshot5/{images,labels}/{train,val,test}/, meta.jsonl, data.yaml,
      targets.json, previews/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rareobj import paths  # noqa: E402
from rareobj.cctv_faceswap import SCRFD, InSwapper, nms  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

CCTV = paths.DATASET / "CCTV"
WORK = paths.DATASET / "CCTV_work"
OUT = WORK / "fewshot5"
V1, F10 = paths.DATASET / "faces_v1", paths.DATASET / "faces10"
PHOTOS = ["IMG-43926.png", "IMG-43392.png", "IMG-42087.png", "IMG-41852.png", "IMG-40798.png"]   # cam 6,20,3,13,5
VAL_CAMS = {19, 22}
N_VAR = 20
N_TARGET = 10
UNK_PER_VAR = 10
SLOT_THR, SLOT_MIN_W = 0.6, 14.0
TARGET_MIN_W = 20.0
SRC_TOP, N_TRY, N_PASS = 20, 4, 2
LABEL_THR = 0.5
SPLIT_N = {"enroll": 5, "swap_train": 70, "swap_test": 25}      # 나머지 = test_hq
EMB_K = 3                     # 스왑 1회에 쓰는 신원 벡터 = 사진 3장 평균 (매번 다른 조합 → 모습 다양성)
EVAL_STRIDE, EVAL_MAX, EVAL_TGT, EVAL_UNK = 5, 400, 3, 3
JPEG_Q = 95
UNKNOWN = 10


def choose_targets(rng) -> list[str]:
    src = np.load(WORK / "swap_sources.npz")
    keep = [str(n) for n, c in zip(src["names"], src["cls"]) if c < UNKNOWN]          # faces10 남성 5명
    pool = sorted(str(n) for n, c in zip(src["names"], src["cls"]) if c == UNKNOWN)   # CLIP 남성
    return keep + [str(n) for n in rng.choice(pool, N_TARGET - len(keep), replace=False)]


def embed_images(det, sw, paths_) -> dict[str, np.ndarray]:
    out = {}
    for p in paths_:
        im = cv2.copyMakeBorder(cv2.imread(str(p)), 80, 80, 80, 80, cv2.BORDER_CONSTANT)
        b, s, k = det.detect(im, scale=1.0, thr=0.5)
        if len(b):
            out[str(p)] = sw.embed(im, k[int(np.argmax(s))])
    return out


def build_identities(det, sw, rng):
    cache = OUT / "identities.json"
    if cache.exists():
        d = json.loads(cache.read_text(encoding="utf-8"))
        E = np.load(OUT / "identity_embs.npy")
        return d, dict(zip(d["emb_keys"], E))
    rows = list(csv.DictReader(open(V1 / "manifest.csv", encoding="utf-8")))
    by_name = {}
    for r in rows:
        by_name.setdefault(r["name"], []).append(V1 / r["path"])
    targets = choose_targets(rng)
    d = {"targets": [], "unknown_train": [], "unknown_test": []}
    all_paths = []
    for c, n in enumerate(targets):
        ps = list(by_name[n])
        rng.shuffle(ps)
        sp, i = {}, 0
        for k, m in SPLIT_N.items():
            sp[k] = [str(p) for p in ps[i:i + m]]
            i += m
        sp["test_hq"] = [str(p) for p in ps[i:]]
        d["targets"].append({"cls": c, "name": n, **sp})
        all_paths += sp["swap_train"] + sp["swap_test"]
    src = np.load(WORK / "swap_sources.npz")
    unk = sorted(str(n) for n, cc in zip(src["names"], src["cls"]) if cc == UNKNOWN and str(n) not in targets)
    rng.shuffle(unk)
    half = len(unk) // 2
    for key, names in (("unknown_train", unk[:half]), ("unknown_test", unk[half:])):
        for n in names:
            ps = [str(p) for p in by_name[n][:EMB_K]]
            d[key].append({"name": n, "imgs": ps})
            all_paths += ps
    print(f"신원 벡터 계산 {len(all_paths)}장", flush=True)
    emb = embed_images(det, sw, all_paths)
    OUT.mkdir(parents=True, exist_ok=True)
    # npz 키에 경로(\, :)를 쓰면 다시 읽을 때 키가 바뀐다 → 키 목록은 json, 벡터는 npy
    d["emb_keys"] = list(emb)
    np.save(OUT / "identity_embs.npy", np.stack(list(emb.values())).astype(np.float32))
    cache.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    return d, emb


def sharpness(p) -> float:
    return float(cv2.Laplacian(cv2.imread(p, cv2.IMREAD_GRAYSCALE), cv2.CV_64F).var())


def nrm(v):
    return v / np.linalg.norm(v)


def select_sources(emb, paths_) -> tuple[list[str], np.ndarray]:
    """흐린 사진을 빼고 신원 중심에 가까운 SRC_TOP 장 + 선택 기준 중심."""
    ps = [p for p in paths_ if p in emb]
    E = np.stack([emb[p] for p in ps])
    cen = nrm(E.mean(0))
    sh = np.array([sharpness(p) for p in ps])
    score = E @ cen - 10.0 * (sh < np.median(sh))
    return [ps[i] for i in np.argsort(-score)[:SRC_TOP]], cen


def mean_emb(emb, paths_, rng, k=EMB_K):
    ps = [p for p in paths_ if p in emb]
    pick = rng.choice(len(ps), min(k, len(ps)), replace=False)
    e = np.mean([emb[ps[i]] for i in pick], 0)
    return e / np.linalg.norm(e)


def slots_of(rec, min_w=SLOT_MIN_W) -> list[int]:
    return [i for i, (b, s, f) in enumerate(zip(rec["boxes"], rec["scores"], rec["frontal"]))
            if s >= SLOT_THR and f and b[2] - b[0] >= min_w]


def split_slots(rec, rng, n_tgt):
    """타깃은 큰 자리(≥TARGET_MIN_W)에서, 나머지 자리는 모르는 사람용으로 섞어 돌려준다."""
    big = [int(i) for i in rng.permutation(slots_of(rec, TARGET_MIN_W))][:n_tgt]
    rest = [int(i) for i in rng.permutation([i for i in slots_of(rec) if i not in big])]
    return big, rest


def _iou(a, B):
    x0, y0 = np.maximum(a[0], B[:, 0]), np.maximum(a[1], B[:, 1])
    x1, y1 = np.minimum(a[2], B[:, 2]), np.minimum(a[3], B[:, 3])
    i = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - i + 1e-9)


def refine_labels(det, img, rec) -> dict:
    """의사 라벨 보강: 2·3배 × 좌우반전 4회 중 2회 이상(평균 점수≥LABEL_THR) 잡혔는데 기존 라벨에 없는 얼굴을 더한다."""
    W = img.shape[1]
    B, S, R = [], [], []
    for j, (sc, fl) in enumerate([(2, False), (2, True), (3, False), (3, True)]):
        b, s, _ = det.detect(img[:, ::-1].copy() if fl else img, scale=sc, thr=0.3)
        if fl:
            b = np.stack([W - b[:, 2], b[:, 1], W - b[:, 0], b[:, 3]], 1)
        B.append(b); S.append(s); R.append(np.full(len(b), j))
    B, S, R = np.concatenate(B), np.concatenate(S), np.concatenate(R)
    cur = np.array([b for b, s in zip(rec["boxes"], rec["scores"]) if s >= LABEL_THR]).reshape(-1, 4)
    extra = []
    for i in nms(B, S, 0.4):
        m = _iou(B[i], B) > 0.4
        if len(set(R[m])) >= 2 and S[m].mean() >= LABEL_THR and (len(cur) == 0 or _iou(B[i], cur).max() < 0.3):
            extra.append([round(float(v), 2) for v in B[i]])
    return {**rec, "extra_boxes": extra}


def yolo_lines(rec) -> list[str]:
    W, H = rec["w"], rec["h"]
    out = []
    extra = [(b, 1.0) for b in rec.get("extra_boxes", [])]
    for (x0, y0, x1, y1), s in list(zip(rec["boxes"], rec["scores"])) + extra:
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        if s >= LABEL_THR and x1 - x0 >= 2 and y1 - y0 >= 2:
            out.append(f"0 {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
    return out


def swap_best(sw, img, kps, emb, src, rng):
    """src = (사진 목록, 선택 기준 중심): 사진 조합 N_TRY 개 × N_PASS 회 스왑 중 중심에 가장 가까운 결과."""
    ps, cen = src
    best = None
    for _ in range(N_TRY):
        e, out = mean_emb(emb, ps, rng), img
        for _ in range(N_PASS):
            out, _ = sw.swap(out, kps, e)
        s = float(sw.embed(out, kps) @ cen)
        if best is None or s > best[0]:
            best = (s, out)
    return best[1]


def compose(sw, img, rec, assign, emb, rng):
    """assign: [(얼굴 번호, cls, 이름, (사진 목록, 선택 중심), 평가 중심)] → 스왑한 프레임과 얼굴별 기록.
    id_cos 는 평가 중심 기준 — 타깃은 선택에 안 쓴 등록 사진 중심."""
    out, faces = img.copy(), []
    for i, c, name, src, held in assign:
        kps = np.float32(rec["kps"][i])
        out = swap_best(sw, out, kps, emb, src, rng)
        faces.append({"face": int(i), "box": rec["boxes"][i], "kps": rec["kps"][i], "cls": c, "who": name,
                      "id_cos": round(float(sw.embed(out, kps) @ held), 3)})
    return out, faces


def save(split, name, im, rec, faces, src_frame, meta, st):
    cv2.imwrite(str(OUT / "images" / split / f"{name}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
    lines = yolo_lines(rec)
    (OUT / "labels" / split / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
    meta.write(json.dumps({"image": f"images/{split}/{name}.jpg", "split": split, "src_frame": src_frame,
                           "cam": rec["cam"], "n_faces": len(lines), "n_extra": len(rec.get("extra_boxes", [])),
                           "swaps": faces}) + "\n")
    st[split] = st.get(split, 0) + 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-var", type=int, default=N_VAR)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    recs = {json.loads(l)["file"]: json.loads(l) for l in open(CCTV / "pseudo_labels.jsonl", encoding="utf-8")}
    det, sw = SCRFD(), InSwapper()
    ids, emb = build_identities(det, sw, rng)
    T = ids["targets"]
    print("타깃:", [(t["cls"], t["name"]) for t in T], flush=True)
    enr = embed_images(det, sw, [p for t in T for p in t["enroll"]])
    for t in T:
        t["held"] = nrm(np.mean([enr[p] for p in t["enroll"] if p in enr], 0))
        t["src_train"] = select_sources(emb, t["swap_train"])
        t["src_test"] = select_sources(emb, t["swap_test"])

    def unk_src(u):
        ps = [p for p in u["imgs"] if p in emb]
        return ps, nrm(np.mean([emb[p] for p in ps], 0))
    for sp in ("train", "val", "test"):
        (OUT / "images" / sp).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / sp).mkdir(parents=True, exist_ok=True)
    (OUT / "previews").mkdir(exist_ok=True)
    meta = open(OUT / "meta.jsonl", "w", encoding="utf-8")
    st, t0 = {}, time.time()
    cos = {"train": [], "test": []}

    # 학습: 사진 5장 × N_VAR 변형
    photo_cams = set()
    for f in PHOTOS:
        img = cv2.imread(str(CCTV / "RawDataFrames" / f))
        rec = refine_labels(det, img, recs[f])
        photo_cams.add(rec["cam"])
        slots = slots_of(rec)
        save("train", Path(f).stem + "_orig", img, rec, [], f, meta, st)          # 스왑 없는 원본도 1장
        for v in range(args.n_var):
            tgt, rest = split_slots(rec, rng, N_TARGET)
            assign = [(tgt[c], t["cls"], t["name"], t["src_train"], t["held"]) for c, t in enumerate(T)]
            for i in rest[:UNK_PER_VAR]:
                u = ids["unknown_train"][int(rng.integers(len(ids["unknown_train"])))]
                s = unk_src(u)
                assign.append((i, UNKNOWN, u["name"], s, s[1]))
            out, faces = compose(sw, img, rec, assign, emb, rng)
            cos["train"] += [x["id_cos"] for x in faces if x["cls"] < UNKNOWN]
            save("train", f"{Path(f).stem}_v{v:02d}", out, rec, faces, f, meta, st)
        print(f"  {f} cam{rec['cam']} 자리 {len(slots)}  {time.time() - t0:.0f}s", flush=True)

    # val/test: 사진 5장의 카메라를 뺀 나머지 카메라
    for sp in ("val", "test"):
        rs = [r for r in recs.values() if r["cam"] not in photo_cams
              and (r["cam"] in VAL_CAMS) == (sp == "val")]
        rs = sorted(rs, key=lambda r: r["idx"])[::EVAL_STRIDE]
        rs = [r for r in rs if len(slots_of(r, TARGET_MIN_W)) >= EVAL_TGT
              and len(slots_of(r)) >= EVAL_TGT + EVAL_UNK][:EVAL_MAX]
        for r in rs:
            img = cv2.imread(str(CCTV / "RawDataFrames" / r["file"]))
            r = refine_labels(det, img, r)
            tgt, rest = split_slots(r, rng, EVAL_TGT)
            tg = rng.choice(len(T), EVAL_TGT, replace=False)
            assign = [(tgt[k], T[c]["cls"], T[c]["name"], T[c]["src_test"], T[c]["held"]) for k, c in enumerate(tg)]
            for i in rest[:EVAL_UNK]:
                u = ids["unknown_test"][int(rng.integers(len(ids["unknown_test"])))]
                s = unk_src(u)
                assign.append((i, UNKNOWN, u["name"], s, s[1]))
            out, faces = compose(sw, img, r, assign, emb, rng)
            cos["test"] += [x["id_cos"] for x in faces if x["cls"] < UNKNOWN]
            save(sp, Path(r["file"]).stem, out, r, faces, r["file"], meta, st)
        print(f"  [{sp}] 프레임 {len(rs)} 카메라 {sorted({r['cam'] for r in rs})}  {time.time() - t0:.0f}s", flush=True)
    meta.close()
    for t in T:
        t["src_train"], t["src_test"] = t["src_train"][0], t["src_test"][0]      # 고른 사진 목록만 남긴다
        del t["held"]
    (OUT / "targets.json").write_text(json.dumps(T, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "data.yaml").write_text(f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\n"
                                   f"test: images/test\nnames:\n  0: face\n", encoding="utf-8")
    print("[done]", st, {k: f"med {np.median(v):.3f}" for k, v in cos.items() if v}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
