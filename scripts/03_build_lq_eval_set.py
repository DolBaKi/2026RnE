"""real_val 에 열화 파이프라인을 적용해 real_val_lq(저화질 평가셋)를 만든다.

1차에서 확인한 문제: COCO val2017의 bear 박스 71개 중 small 은 5개뿐이라
"저화질 곰 검출 성능"을 잴 방법이 없었다. 그 자체로는 새 이미지가 필요하지만,
해상도를 낮추는 것은 별개 문제다 — 실제 이미지는 그대로 두고 촬영 조건만
저화질로 바꾼 쌍(real_val ↔ real_val_lq)을 만들면 **동일한 71개 박스**에 대해
"화질만 낮아졌을 때 사전학습 YOLO 성능이 얼마나 떨어지는가"를 정량화할 수 있다.
이는 2차 학습에 쓰이지 않는 순수 평가용 산출물이다.

라벨(bbox)은 열화가 리사이즈 왕복(다운스케일 후 원 크기로 업샘플)만 하고 캔버스
크기 자체는 바꾸지 않으므로 원본 YOLO 라벨을 그대로 복사한다.

출력
  data/interim/real_bear/real_val_lq/images/*.jpg
  data/interim/real_bear/real_val_lq/labels/*.txt   (real_val 라벨과 동일 좌표)
  data/interim/real_bear/real_val_lq/degrade_params.json
  reports/figures/fig04_degradation_examples.png
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rareobj import degrade, paths, viz  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

SEED = 20260819  # 오늘 세션 고정 시드 — 재현성
SRC_IMG = paths.RAW / "coco" / "images" / "real_val"
SRC_LBL = paths.INTERIM / "real_bear" / "real_val" / "labels"
DST = paths.INTERIM / "real_bear" / "real_val_lq"


def main() -> int:
    if not SRC_IMG.exists():
        raise SystemExit(f"먼저 scripts/02_build_real_bear_sets.py 실행: {SRC_IMG} 없음")

    rng = random.Random(SEED)
    (DST / "images").mkdir(parents=True, exist_ok=True)
    (DST / "labels").mkdir(parents=True, exist_ok=True)

    files = sorted(SRC_IMG.glob("*.jpg"))
    records = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"[skip] 디코드 실패: {f.name}")
            continue
        lq, params = degrade.degrade_image(img, rng)
        cv2.imwrite(str(DST / "images" / f.name), lq)
        shutil.copy(SRC_LBL / f"{f.stem}.txt", DST / "labels" / f"{f.stem}.txt")
        records.append({"file_name": f.name, **params.to_dict()})

    (DST / "degrade_params.json").write_text(
        json.dumps({"seed": SEED, "ranges": degrade.RANGES, "per_image": records},
                  ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[done] {len(records)}장 열화 완료 -> {DST}")

    # 시각 대조 figure: 원본 vs 열화, 3쌍 무작위 샘플
    viz.use_theme()
    sample = rng.sample(records, min(3, len(records)))
    fig, axes = plt.subplots(2, len(sample), figsize=(3.6 * len(sample), 7.4))
    for j, rec in enumerate(sample):
        name = rec["file_name"]
        orig = cv2.cvtColor(cv2.imread(str(SRC_IMG / name)), cv2.COLOR_BGR2RGB)
        lq = cv2.cvtColor(cv2.imread(str(DST / "images" / name)), cv2.COLOR_BGR2RGB)
        axes[0, j].imshow(orig)
        axes[0, j].set_title(name, fontsize=8, color=viz.INK_2)
        axes[1, j].imshow(lq)
        cap = (f"blur {rec['motion_blur_len']}px  ↓×{rec['downscale_factor']:.2f}  "
               f"σ{rec['noise_sigma']:.1f}  jpg-Q{rec['jpeg_quality']}")
        axes[1, j].set_xlabel(cap, fontsize=7.5, color=viz.INK_2)
        for ax in (axes[0, j], axes[1, j]):
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
    axes[0, 0].set_ylabel("원본 (real_val)", fontsize=9, color=viz.INK, fontweight="bold")
    axes[1, 0].set_ylabel("열화 적용 (real_val_lq)", fontsize=9, color=viz.INK, fontweight="bold")
    fig.suptitle("저화질 감시영상 열화 파이프라인 — 모션블러 → 다운스케일 → 센서노이즈 → JPEG",
                x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = paths.FIGURES / "fig04_degradation_examples.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[figure] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
