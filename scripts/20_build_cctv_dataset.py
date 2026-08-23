"""6차 — 저화질 CCTV 인물/소지품/차량 탐지 파일럿.

배경. bear 프로젝트(1~5차)와 달리 이번 8개 클래스는 COCO에서 전혀 희소하지
않다 — "부족한 데이터를 생성모델로 보완"할 필요가 없다. 대신 연구 질문은
"화질이 나쁜 CCTV에서도 사람·소지품·차량을 놓치지 않는가"로, 5차에서 이미
검증된 두 가지를 그대로 재사용한다: ① 원본+Albumentations 열화를 함께 학습에
쓰는 강건화, ② NAFNet+Real-ESRGAN 복원 전처리. 평가는 5차의 핵심 교훈대로
반드시 "우리가 흐리게 만든 사진"이 아니라 COCO에 원래부터 있는 자연 저화질
사진으로 한다(scripts/21).

OD-VIRAT(실제 CCTV 전용 벤치마크, Person/Car/Vehicle/Carrying_object/Bike의
5클래스)를 조사했으나 Google Drive 폴더 배포라 스크립트 재현이 불안정해
보류했다 — 링크는 docs/05_cctv_pipeline.md에 남겨 다음 시도로 미룬다. 대신
이미 로컬에 있는 COCO train2017 어노테이션과 검증된 다운로드 파이프라인을
그대로 써서 사람+동행 소지품/이동수단 조합의 "거리·감시 장면"에 해당하는
이미지를 뽑는다.

클래스 (COCO index 그대로 사용 — bear처럼 80클래스 헤드를 유지해 baseline과
동일 잣대로 비교 가능하게 한다)
  0 person, 1 bicycle, 2 car, 3 motorcycle, 5 bus,
  24 backpack, 26 handbag, 28 suitcase

출력
  data/dataset/cctv_synth/images/{train,val}/*.jpg   (clean + Albumentations열화 쌍)
  data/dataset/cctv_synth/labels/{train,val}/*.txt
  data/dataset/cctv_synth/manifest.jsonl
  data/interim/cctv_pool/                            (real_lowq 구축용 별도 홀드아웃 풀)
"""

from __future__ import annotations

import json
import os
import random
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rareobj import albu_degrade, paths  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

TARGET_NAMES = ["person", "bicycle", "car", "motorcycle", "bus",
                "backpack", "handbag", "suitcase"]
SEED = 20260823
N_TRAIN = 420      # clean 이미지 수 (실제 학습셋은 clean+deg 쌍이라 2배)
N_VAL = 40
N_POOL = 220       # real_lowq 선정용 별도 홀드아웃 풀 (학습에 안 씀)
OUT = paths.DATASET / "cctv_synth"
POOL_DIR = paths.INTERIM / "cctv_pool"


def coco_index_map() -> dict[str, int]:
    import ultralytics.cfg as C

    p = Path(C.__file__).parent / "datasets" / "coco.yaml"
    names = yaml.safe_load(p.read_text(encoding="utf-8"))["names"]
    return {v: k for k, v in names.items()}


def fetch(url: str, dst: Path) -> str:
    if dst.exists() and dst.stat().st_size > 1024:
        return "skip"
    req = urllib.request.Request(url.replace("https://", "http://"),
                                 headers={"User-Agent": "curl/8"})
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = resp.read()
    except (urllib.error.URLError, TimeoutError, OSError):
        return "fail"
    if len(data) < 1024:
        return "fail"
    dst.write_bytes(data)
    return "ok"


def write_yolo_labels(anns: list[dict], W: int, H: int, path: Path, catid_to_yolo: dict[int, int]) -> None:
    """anns의 category_id는 COCO 원본 JSON의 id(1,2,3,4,6,27,31,33 ...)다 —
    ultralytics coco.yaml의 연속 0-79 인덱스와 다르다(이 둘을 헷갈려서 처음에
    라벨을 통째로 잘못 쓴 적이 있다: baseline mAP가 CLEAN 이미지에서도 0.001로
    나와서 발견함 — 실제로는 검출기가 아니라 라벨 인덱스가 틀렸던 것).
    반드시 catid_to_yolo로 변환한 값을 써야 한다."""
    lines = []
    for a in anns:
        x, y, w, h = a["bbox"]
        cx, cy = (x + w / 2) / W, (y + h / 2) / H
        yolo_idx = catid_to_yolo[a["category_id"]]
        lines.append(f"{yolo_idx} {cx:.6f} {cy:.6f} {w/W:.6f} {h/H:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> int:
    paths.ensure()
    name_to_idx = coco_index_map()  # 이름 -> ultralytics 0-79 연속 인덱스

    ann_path = paths.COCO_ANN / "instances_train2017.json"
    print(f"[load] {ann_path.name} …")
    with open(ann_path, encoding="utf-8") as f:
        ann = json.load(f)

    # COCO 원본 JSON의 category_id(1,2,3,4,6,27,31,33 ...)는 ultralytics의
    # 연속 0-79 인덱스와 다르다 — 이름을 거쳐 두 공간을 명시적으로 이어준다.
    catid_to_name = {c["id"]: c["name"] for c in ann["categories"]}
    catid_to_yolo = {cid: name_to_idx[name] for cid, name in catid_to_name.items()
                     if name in TARGET_NAMES}
    target_raw_ids = set(catid_to_yolo)
    print(f"[map] COCO raw id -> yolo idx: "
         f"{ {catid_to_name[c]: (c, y) for c, y in catid_to_yolo.items()} }")

    imgs = {i["id"]: i for i in ann["images"]}
    by_img: dict[int, list[dict]] = {}
    for a in ann["annotations"]:
        if a["category_id"] in target_raw_ids and not a.get("iscrowd", 0):
            by_img.setdefault(a["image_id"], []).append(a)

    # "감시 장면"다운 이미지를 우선한다 — 관련 클래스가 여러 개 같이 등장하는 사진
    # (사람+가방, 사람+자전거 등)이 CCTV 맥락에 더 가깝다.
    scored = sorted(by_img.items(),
                    key=lambda kv: len({a["category_id"] for a in kv[1]}), reverse=True)
    rng = random.Random(SEED)
    pool_ids = [iid for iid, _ in scored]
    rng.shuffle(pool_ids[len(pool_ids) // 3:])  # 상위 1/3(다중클래스)은 순서 유지, 나머지는 섞어 다양성 확보

    need = N_TRAIN + N_VAL + N_POOL
    picked = pool_ids[:need]
    train_ids, val_ids, pool_ids_final = picked[:N_TRAIN], picked[N_TRAIN:N_TRAIN + N_VAL], picked[N_TRAIN + N_VAL:]
    print(f"[select] train {len(train_ids)} / val {len(val_ids)} / pool(real_lowq용) {len(pool_ids_final)}")

    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
    (POOL_DIR / "images").mkdir(parents=True, exist_ok=True)
    (POOL_DIR / "labels").mkdir(parents=True, exist_ok=True)

    def download_group(ids: list[int], img_dir: Path) -> dict[int, str]:
        status = {}
        with ThreadPoolExecutor(max_workers=16) as ex:
            futs = {ex.submit(fetch, imgs[iid]["coco_url"], img_dir / imgs[iid]["file_name"]): iid
                   for iid in ids}
            for i, fut in enumerate(as_completed(futs), 1):
                iid = futs[fut]
                status[iid] = fut.result()
                if i % 50 == 0 or i == len(ids):
                    print(f"\r  {i}/{len(ids)}", end="")
        print()
        return status

    tmp_dl = paths.INTERIM / "cctv_download_tmp"
    tmp_dl.mkdir(parents=True, exist_ok=True)
    print("[download] 대상 이미지…")
    status = download_group(train_ids + val_ids + pool_ids_final, tmp_dl)
    print(f"[download] 성공 {sum(1 for v in status.values() if v=='ok')} / 시도 {len(status)}")

    pipe = albu_degrade.build_pipeline()
    manifest = []

    def process(iid: int, split: str) -> bool:
        im = imgs[iid]
        src = tmp_dl / im["file_name"]
        if not src.exists():
            return False
        img = cv2.imread(str(src))
        if img is None:
            return False
        H, W = img.shape[:2]
        anns = [a for a in by_img[iid] if a["category_id"] in target_raw_ids]
        stem = Path(im["file_name"]).stem

        if split == "pool":
            cv2.imwrite(str(POOL_DIR / "images" / im["file_name"]), img)
            write_yolo_labels(anns, W, H, POOL_DIR / "labels" / f"{stem}.txt", catid_to_yolo)
            return True

        # clean
        cv2.imwrite(str(OUT / "images" / split / f"{stem}_clean.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        write_yolo_labels(anns, W, H, OUT / "labels" / split / f"{stem}_clean.txt", catid_to_yolo)
        # degraded (bbox는 리사이즈 왕복만 하므로 좌표 불변 — 같은 라벨 재사용)
        deg = albu_degrade.degrade_bgr(img, pipe)
        cv2.imwrite(str(OUT / "images" / split / f"{stem}_deg.jpg"), deg, [cv2.IMWRITE_JPEG_QUALITY, 90])
        write_yolo_labels(anns, W, H, OUT / "labels" / split / f"{stem}_deg.txt", catid_to_yolo)

        manifest.append({"image_id": iid, "stem": stem, "split": split,
                         "n_boxes": len(anns),
                         "classes": sorted({catid_to_yolo[a["category_id"]] for a in anns})})
        return True

    print("[compose] clean+deg 쌍 생성 …")
    for iid in train_ids:
        process(iid, "train")
    for iid in val_ids:
        process(iid, "val")
    for iid in pool_ids_final:
        process(iid, "pool")

    (OUT / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in manifest), encoding="utf-8")

    names = {i: n for n, i in name_to_idx.items()}
    dataset_yaml = {"path": str(OUT).replace("\\", "/"),
                    "train": "images/train", "val": "images/val", "names": names}
    (OUT / "dataset.yaml").write_text(yaml.safe_dump(dataset_yaml, allow_unicode=True, sort_keys=False),
                                      encoding="utf-8")

    n_train_pairs = len(list((OUT / "images" / "train").glob("*_clean.jpg")))
    n_pool = len(list((POOL_DIR / "images").glob("*.jpg")))
    print(f"[done] train {n_train_pairs}쌍(clean+deg={n_train_pairs*2}장) / "
         f"val {len(list((OUT/'images'/'val').glob('*_clean.jpg')))}쌍 / pool {n_pool}장")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
