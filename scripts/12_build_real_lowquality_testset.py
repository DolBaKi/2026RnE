"""COCO 실사진(real_val+real_ref, 449장) 중 원래부터 저화질/블러인 것만 골라
'진짜' 저화질 평가셋(real_lowq)을 만든다.

3.3 성능평가 방법의 핵심 요구사항: "학습에 사용한 인위적 열화 이미지가 아니라
원래부터 화질이 낮거나 블러가 존재하는 실제 저화질 사진을 시험 데이터로 사용".
지금까지 쓴 `real_val_lq`는 깨끗한 실사진에 우리가 만든 열화를 입힌 것이라
이 요구를 충족하지 못한다 — 우리 열화 모델의 특성에 유리하게 편향될 위험이
있기 때문이다. COCO 사진 자체는 촬영 조건이 제각각이라(줌 부족, 손떨림,
저해상도 업로드 등) 그중 자연적으로 저화질인 것만 추리면 열화 시뮬레이션을
전혀 거치지 않은 순수한 실제 저화질 샘플이 된다.

측정 지표 (박스 크롭 기준, 스케일 불변성을 위해 crop을 표준 크기로 리사이즈 후 계산)
  blur_score   Laplacian 분산 — 낮을수록 흐림
  native_px    원본 bbox 짧은 변 픽셀 수 — 작을수록 실제 저해상도

선정: blur_score 하위 30% 또는 native_px 하위 30% 인 인스턴스를 가진 이미지.
이 449장은 전부 생성 파이프라인의 학습 입력으로 쓰인 적이 없다(1차 문서 §1.5 —
real_ref는 프롬프트 설계 참고용으로 '눈으로만' 봤을 뿐 모델에 입력되지 않음).
따라서 평가셋으로 그대로 써도 학습-평가 누수가 없다.

출력
  data/interim/real_lowq/images/*.jpg  (원본 그대로 복사)
  data/interim/real_lowq/labels/*.txt  (YOLO, class 0 = bear)
  data/interim/real_lowq/manifest.json
  reports/real_lowq_selection.csv
  reports/figures/fig10_real_lowq_examples.png
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rareobj import paths, viz  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

OUT = paths.INTERIM / "real_lowq"
BLUR_RESIZE = 224          # crop을 이 짧은변으로 리사이즈 후 라플라시안 분산 계산 (스케일 불변)
QUANTILE = 0.30             # blur 하위 30% 또는 native_px 하위 30%


def laplacian_var(gray_crop: np.ndarray) -> float:
    h, w = gray_crop.shape
    scale = BLUR_RESIZE / min(h, w)
    resized = cv2.resize(gray_crop, (max(1, int(w * scale)), max(1, int(h * scale))))
    return float(cv2.Laplacian(resized, cv2.CV_64F).var())


def collect_instances() -> list[dict]:
    sets = [
        ("real_val", paths.RAW / "coco" / "images" / "real_val",
         paths.INTERIM / "real_bear" / "real_val" / "labels"),
        ("real_ref", paths.RAW / "coco" / "images" / "real_ref",
         paths.INTERIM / "real_bear" / "real_ref" / "labels"),
    ]
    rows = []
    for src_name, img_dir, lbl_dir in sets:
        for lbl_path in sorted(lbl_dir.glob("*.txt")):
            img_path = img_dir / f"{lbl_path.stem}.jpg"
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            H, W = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lines = [l.split() for l in lbl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            for cls, cx, cy, w, h in lines:
                cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
                x0, y0 = max(int(cx - w / 2), 0), max(int(cy - h / 2), 0)
                x1, y1 = min(int(cx + w / 2), W), min(int(cy + h / 2), H)
                if x1 - x0 < 8 or y1 - y0 < 8:
                    continue
                crop = gray[y0:y1, x0:x1]
                rows.append({
                    "source": src_name, "file_name": img_path.name, "stem": img_path.stem,
                    "img_dir": str(img_dir), "lbl_dir": str(lbl_dir),
                    "bbox_xywh": [x0, y0, x1 - x0, y1 - y0],
                    "native_px": min(x1 - x0, y1 - y0),
                    "blur_score": laplacian_var(crop),
                })
    return rows


def main() -> int:
    paths.ensure()
    rows = collect_instances()
    print(f"[collect] {len(rows)}개 bear 인스턴스 (real_val+real_ref)")

    blur_vals = np.array([r["blur_score"] for r in rows])
    px_vals = np.array([r["native_px"] for r in rows])
    blur_thr = float(np.quantile(blur_vals, QUANTILE))
    px_thr = float(np.quantile(px_vals, QUANTILE))
    print(f"[threshold] blur_score <= {blur_thr:.1f} (하위 {QUANTILE:.0%})  "
         f"native_px <= {px_thr:.1f}px (하위 {QUANTILE:.0%})")

    for r in rows:
        r["is_blurry"] = r["blur_score"] <= blur_thr
        r["is_lowres"] = r["native_px"] <= px_thr
        r["is_lowq"] = r["is_blurry"] or r["is_lowres"]

    lowq_files = sorted({(r["source"], r["file_name"]) for r in rows if r["is_lowq"]})
    print(f"[selected] {len(lowq_files)}장 (인스턴스 {sum(r['is_lowq'] for r in rows)}개)")

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    by_file: dict[tuple, list[dict]] = {}
    for r in rows:
        by_file.setdefault((r["source"], r["file_name"]), []).append(r)

    for src, fname in lowq_files:
        recs = by_file[(src, fname)]
        img_dir, lbl_dir = Path(recs[0]["img_dir"]), Path(recs[0]["lbl_dir"])
        shutil.copy(img_dir / fname, OUT / "images" / fname)
        shutil.copy(lbl_dir / f"{recs[0]['stem']}.txt", OUT / "labels" / f"{recs[0]['stem']}.txt")

    manifest = {
        "n_instances_total": len(rows), "n_images_selected": len(lowq_files),
        "blur_threshold_q30": blur_thr, "native_px_threshold_q30": px_thr,
        "criterion": "blur_score <= q30 OR native_px <= q30 (인스턴스 단위, 이미지에 하나라도 해당하면 선정)",
        "selected": [{"source": s, "file_name": f} for s, f in lowq_files],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    import csv
    with open(paths.REPORTS / "real_lowq_selection.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["source", "file_name", "native_px", "blur_score",
                                         "is_blurry", "is_lowres", "is_lowq"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})

    # 산점도 + 선정 임계값 시각화
    viz.use_theme()
    fig, ax = plt.subplots(figsize=(7, 5.2))
    sel = np.array([r["is_lowq"] for r in rows])
    ax.scatter(px_vals[~sel], blur_vals[~sel], s=16, color=viz.BLUE, alpha=0.6, label="제외 (충분히 선명/고해상도)")
    ax.scatter(px_vals[sel], blur_vals[sel], s=20, color=viz.CRITICAL, alpha=0.8, label="real_lowq 로 선정")
    ax.axvline(px_thr, color=viz.MUTED, ls=":", lw=1.2)
    ax.axhline(blur_thr, color=viz.MUTED, ls=":", lw=1.2)
    ax.set_xlabel("bbox 짧은 변 (native px)")
    ax.set_ylabel(f"Laplacian 분산 (crop을 {BLUR_RESIZE}px로 리사이즈 후 측정, 낮을수록 흐림)")
    ax.set_yscale("log")
    ax.set_title("진짜 저화질 실사진 선정 — 자연 발생 블러/저해상도", loc="left", pad=10)
    ax.legend(loc="upper right")
    viz.finish(ax)
    fig.savefig(paths.FIGURES / "fig10_real_lowq_examples.png")
    plt.close(fig)

    print(f"[write] {OUT/'manifest.json'}")
    print(f"[write] {paths.REPORTS/'real_lowq_selection.csv'}")
    print(f"[figure] {paths.FIGURES/'fig10_real_lowq_examples.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
