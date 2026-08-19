"""소규모 다중 클래스 COCO 표본 — 파인튜닝 시 기존 79개 클래스 성능(재앙적 망각)
확인/방지용.

`bear` 데이터만으로 파인튜닝하면 검출 헤드가 "새 클래스 하나"에 치우쳐 학습되면서
기존 79개 클래스 성능이 무너질 수 있다. 이를 확인하려면 여러 클래스가 섞인 소규모
실제 COCO 표본이 필요하다 — 학습에는 낮은 비중으로 섞어 정칙화 역할을 하고, 평가
시에는 "곰을 넣었더니 다른 클래스가 얼마나 나빠졌는가"를 재는 기준이 된다.

전체 train2017 이미지(19GB)를 받을 필요는 없다. 다양성이 중요하므로 여러
supercategory에서 고르게 뽑는다. 라벨은 COCO 80클래스 인덱싱(YOLO 표준 순서)을
그대로 써서 합성 bear 데이터(class 21)와 같은 라벨 공간에서 섞을 수 있게 한다.

출력
  data/interim/reg_sample/images/*.jpg
  data/interim/reg_sample/labels/*.txt   (COCO 80클래스 인덱스, YOLO 좌표)
  data/interim/reg_sample/manifest.json
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rareobj import paths  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

OUT = paths.INTERIM / "reg_sample"
N_IMAGES = 350
SEED = 20260822


def coco80_name_to_index() -> dict[str, int]:
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


def main() -> int:
    import random

    ann_path = paths.COCO_ANN / "instances_train2017.json"
    print(f"[load] {ann_path.name} …")
    with open(ann_path, encoding="utf-8") as f:
        ann = json.load(f)

    name_to_yolo = coco80_name_to_index()
    cat_id_to_name = {c["id"]: c["name"] for c in ann["categories"]}
    bear_cat_id = next(k for k, v in cat_id_to_name.items() if v == "bear")

    imgs = {i["id"]: i for i in ann["images"]}
    by_img: dict[int, list] = {}
    for a in ann["annotations"]:
        if a["category_id"] == bear_cat_id or a.get("iscrowd", 0):
            continue  # bear는 합성 데이터가 담당 — 여기 섞으면 실데이터/생성데이터가 뒤섞인다
        by_img.setdefault(a["image_id"], []).append(a)

    # supercategory별로 고르게 셔플 후 라운드로빈 추출 → 다양성 확보
    cats = {c["id"]: c for c in ann["categories"]}
    super_groups: dict[str, list[int]] = {}
    for iid, alist in by_img.items():
        sc = cats[alist[0]["category_id"]]["supercategory"]
        super_groups.setdefault(sc, []).append(iid)

    rng = random.Random(SEED)
    for g in super_groups.values():
        rng.shuffle(g)

    picked: list[int] = []
    keys = sorted(super_groups)
    ptr = {k: 0 for k in keys}
    while len(picked) < N_IMAGES:
        progressed = False
        for k in keys:
            if ptr[k] < len(super_groups[k]):
                picked.append(super_groups[k][ptr[k]])
                ptr[k] += 1
                progressed = True
                if len(picked) >= N_IMAGES:
                    break
        if not progressed:
            break

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)

    print(f"[fetch] {len(picked)}장 다운로드 (supercategory {len(keys)}종 균형 추출)")
    records = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = {}
        for iid in picked:
            im = imgs[iid]
            dst = OUT / "images" / im["file_name"]
            futs[ex.submit(fetch, im["coco_url"], dst)] = (iid, im)
        done = 0
        for fut in as_completed(futs):
            iid, im = futs[fut]
            status = fut.result()
            done += 1
            if done % 50 == 0:
                print(f"\r  {done}/{len(picked)}", end="")
            if status == "fail":
                continue
            lines = []
            for a in by_img[iid]:
                name = cat_id_to_name[a["category_id"]]
                yidx = name_to_yolo[name]
                x, y, w, h = a["bbox"]
                W, H = im["width"], im["height"]
                cx, cy = (x + w / 2) / W, (y + h / 2) / H
                lines.append(f"{yidx} {cx:.6f} {cy:.6f} {w/W:.6f} {h/H:.6f}")
            stem = Path(im["file_name"]).stem
            (OUT / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            records.append({"image_id": iid, "file_name": im["file_name"],
                           "n_boxes": len(lines),
                           "supercategory": cats[by_img[iid][0]["category_id"]]["supercategory"]})
    print()

    (OUT / "manifest.json").write_text(
        json.dumps({"n_images": len(records), "seed": SEED, "images": records},
                  ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    sc_count: dict[str, int] = {}
    for r in records:
        sc_count[r["supercategory"]] = sc_count.get(r["supercategory"], 0) + 1
    print(f"[done] {len(records)}장 확보 / supercategory 분포 {sc_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
