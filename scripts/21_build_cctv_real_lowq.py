"""cctv_pool(학습에 전혀 안 쓰인 홀드아웃 220장)에서 자연 발생 저화질/블러
인스턴스만 골라 real_lowq_cctv를 만든다. 방법은 scripts/12(bear)와 동일 —
5차의 핵심 교훈(우리가 흐리게 만든 사진이 아니라 원래부터 나쁜 화질인 실사진으로
평가해야 한다)을 이번에도 그대로 적용한다.

출력
  data/interim/real_lowq_cctv/images/*.jpg
  data/interim/real_lowq_cctv/labels/*.txt   (COCO 인덱스 그대로)
  data/interim/real_lowq_cctv/manifest.json
  reports/figures/fig14_cctv_real_lowq.png
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

POOL = paths.INTERIM / "cctv_pool"
OUT = paths.INTERIM / "real_lowq_cctv"
BLUR_RESIZE = 224
QUANTILE = 0.35   # pool 자체가 220장으로 작아 bear(0.30)보다 살짝 넉넉하게 잡는다


def laplacian_var(gray_crop: np.ndarray) -> float:
    h, w = gray_crop.shape
    scale = BLUR_RESIZE / min(h, w)
    resized = cv2.resize(gray_crop, (max(1, int(w * scale)), max(1, int(h * scale))))
    return float(cv2.Laplacian(resized, cv2.CV_64F).var())


def main() -> int:
    paths.ensure()
    rows = []
    for lbl_path in sorted((POOL / "labels").glob("*.txt")):
        img_path = POOL / "images" / f"{lbl_path.stem}.jpg"
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
            rows.append({"file_name": img_path.name, "stem": img_path.stem, "cls": int(cls),
                        "native_px": min(x1 - x0, y1 - y0), "blur_score": laplacian_var(crop)})

    print(f"[collect] {len(rows)}개 인스턴스 ({len({r['file_name'] for r in rows})}장)")
    blur_vals = np.array([r["blur_score"] for r in rows])
    px_vals = np.array([r["native_px"] for r in rows])
    blur_thr = float(np.quantile(blur_vals, QUANTILE))
    px_thr = float(np.quantile(px_vals, QUANTILE))
    print(f"[threshold] blur<={blur_thr:.1f}  native_px<={px_thr:.1f}")

    for r in rows:
        r["is_lowq"] = (r["blur_score"] <= blur_thr) or (r["native_px"] <= px_thr)

    lowq_files = sorted({r["file_name"] for r in rows if r["is_lowq"]})
    print(f"[selected] {len(lowq_files)}장")

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    for fname in lowq_files:
        stem = Path(fname).stem
        shutil.copy(POOL / "images" / fname, OUT / "images" / fname)
        shutil.copy(POOL / "labels" / f"{stem}.txt", OUT / "labels" / f"{stem}.txt")

    manifest = {"n_instances": len(rows), "n_images_selected": len(lowq_files),
               "blur_threshold": blur_thr, "native_px_threshold": px_thr,
               "selected": lowq_files}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    viz.use_theme()
    sel = np.array([r["is_lowq"] for r in rows])
    fig, ax = plt.subplots(figsize=(7, 5.2))
    ax.scatter(px_vals[~sel], blur_vals[~sel], s=16, color=viz.BLUE, alpha=0.6, label="제외")
    ax.scatter(px_vals[sel], blur_vals[sel], s=20, color=viz.CRITICAL, alpha=0.8, label="real_lowq_cctv 선정")
    ax.axvline(px_thr, color=viz.MUTED, ls=":", lw=1.2)
    ax.axhline(blur_thr, color=viz.MUTED, ls=":", lw=1.2)
    ax.set_xlabel("bbox 짧은 변 (native px)")
    ax.set_ylabel("Laplacian 분산 (낮을수록 흐림)")
    ax.set_yscale("log")
    ax.set_title("CCTV 파일럿 — 진짜 저화질 실사진 선정", loc="left", pad=10)
    ax.legend(loc="upper right")
    viz.finish(ax)
    fig.savefig(paths.FIGURES / "fig14_cctv_real_lowq.png")
    plt.close(fig)
    print(f"[write] {OUT/'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
