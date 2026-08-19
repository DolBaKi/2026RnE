"""COCO 실제 bear 이미지 → 실데이터 기준셋(YOLO 포맷) 구축.

역할 분담을 분명히 해 둔다. 제출용 '희소 객체 데이터셋'은 3차에서 생성모델로
만드는 쪽이고, 여기서 받는 실제 이미지는 그 데이터셋에 섞지 않는다. 용도는 둘.

  real_val  (val2017 유래)  : 성능 검증용 홀드아웃. 생성 데이터 학습 전/후 비교의
                              유일한 판정 기준. 학습에 절대 넣지 않는다.
  real_ref  (train2017 유래): 생성물 사실성 측정(FID/CLIP)과 프롬프트 설계용 참조.
                              사람이 눈으로 보고 프롬프트를 잡는 데도 쓴다.

출력
  data/raw/coco/images/{real_val,real_ref}/*.jpg
  data/interim/real_bear/{real_val,real_ref}/labels/*.txt   (YOLO: cls cx cy w h, 정규화)
  data/interim/real_bear/manifest.json                      (라이선스·출처·bbox 통계)
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rareobj import paths  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

TARGET = "bear"
IMG_ROOT = paths.RAW / "coco" / "images"
OUT_ROOT = paths.INTERIM / "real_bear"


def collect(split: str, limit: int | None) -> tuple[list[dict], dict]:
    p = paths.COCO_ANN / f"instances_{split}2017.json"
    print(f"[load] {p.name} …")
    with open(p, encoding="utf-8") as f:
        ann = json.load(f)

    cid = next(c["id"] for c in ann["categories"] if c["name"] == TARGET)
    licenses = {l["id"]: l for l in ann["licenses"]}
    imgs = {i["id"]: i for i in ann["images"]}

    per_img: dict[int, list[dict]] = {}
    for a in ann["annotations"]:
        if a["category_id"] != cid or a.get("iscrowd", 0):
            continue
        per_img.setdefault(a["image_id"], []).append(a)

    # 재현성: image_id 오름차순 고정. limit 을 걸어도 항상 같은 부분집합.
    ids = sorted(per_img)
    if limit is not None:
        ids = ids[:limit]

    records = []
    for iid in ids:
        im = imgs[iid]
        boxes = []
        for a in per_img[iid]:
            x, y, w, h = a["bbox"]
            if w <= 1 or h <= 1:
                continue
            boxes.append(
                {
                    "bbox_xywh": [x, y, w, h],
                    "area": a["area"],
                    "size_bin": "small" if a["area"] < 32**2
                    else ("medium" if a["area"] < 96**2 else "large"),
                }
            )
        if not boxes:
            continue
        lic = licenses.get(im["license"], {})
        records.append(
            {
                "image_id": iid,
                "file_name": im["file_name"],
                "width": im["width"],
                "height": im["height"],
                "coco_url": im["coco_url"],
                "flickr_url": im.get("flickr_url", ""),
                "license_id": im["license"],
                "license_name": lic.get("name", ""),
                "license_url": lic.get("url", ""),
                "boxes": boxes,
            }
        )
    return records, {"category_id": cid, "split": f"{split}2017"}


def fetch(rec: dict, dst_dir: Path) -> tuple[str, str]:
    dst = dst_dir / rec["file_name"]
    if dst.exists() and dst.stat().st_size > 1024:
        return rec["file_name"], "skip"
    # https 인증서 호스트명 불일치 회피(00 스크립트와 동일 이유) → http 경로 사용
    url = rec["coco_url"].replace("https://", "http://")
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = resp.read()
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return rec["file_name"], f"fail:{type(e).__name__}"
    if len(data) < 1024:
        return rec["file_name"], "fail:short"
    dst.write_bytes(data)
    return rec["file_name"], "ok"


def write_labels(records: list[dict], label_dir: Path) -> None:
    """YOLO 포맷. 이 기준셋은 bear 단일 클래스이므로 class index 0 고정."""
    label_dir.mkdir(parents=True, exist_ok=True)
    for rec in records:
        W, H = rec["width"], rec["height"]
        lines = []
        for b in rec["boxes"]:
            x, y, w, h = b["bbox_xywh"]
            cx, cy = (x + w / 2) / W, (y + h / 2) / H
            lines.append(f"0 {cx:.6f} {cy:.6f} {w / W:.6f} {h / H:.6f}")
        (label_dir / f"{Path(rec['file_name']).stem}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )


def build(split: str, name: str, limit: int | None, workers: int) -> dict:
    records, meta = collect(split, limit)
    img_dir = IMG_ROOT / name
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{name}] 이미지 {len(records)}장 / bear 박스 "
          f"{sum(len(r['boxes']) for r in records)}개 다운로드")
    stats: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch, r, img_dir): r for r in records}
        for i, fut in enumerate(as_completed(futs), 1):
            _, status = fut.result()
            key = status.split(":")[0]
            stats[key] = stats.get(key, 0) + 1
            if i % 100 == 0 or i == len(records):
                print(f"\r  {i}/{len(records)}  {stats}", end="")
    print()

    ok = [r for r in records if (img_dir / r["file_name"]).exists()]
    write_labels(ok, OUT_ROOT / name / "labels")

    bins: dict[str, int] = {}
    for r in ok:
        for b in r["boxes"]:
            bins[b["size_bin"]] = bins.get(b["size_bin"], 0) + 1
    print(f"[{name}] 확보 {len(ok)}장 / 박스 {sum(bins.values())}개  크기분포={bins}")
    return {
        **meta,
        "name": name,
        "n_images_requested": len(records),
        "n_images_downloaded": len(ok),
        "n_boxes": sum(bins.values()),
        "size_bins": bins,
        "download_status": stats,
        "images": ok,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-limit", type=int, default=400,
                    help="train2017 참조셋 상한 (FID/프롬프트 설계용, 전량 불필요)")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    paths.ensure()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    manifest = {
        "target_class": TARGET,
        "source": "COCO 2017 (cocodataset.org)",
        "note": "real_val 은 홀드아웃 평가 전용 — 학습·생성 파이프라인에 투입 금지",
        "sets": [
            build("val", "real_val", None, args.workers),
            build("train", "real_ref", args.ref_limit, args.workers),
        ],
    }
    out = OUT_ROOT / "manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[write] {out}")

    lic: dict[str, int] = {}
    for s in manifest["sets"]:
        for im in s["images"]:
            k = im["license_name"] or f"id{im['license_id']}"
            lic[k] = lic.get(k, 0) + 1
    print("[라이선스 분포]")
    for k, v in sorted(lic.items(), key=lambda kv: -kv[1]):
        print(f"  {v:>5}  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
