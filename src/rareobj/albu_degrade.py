"""Albumentations 기반 열화 — 모션블러 + 다운스케일.

src/rareobj/degrade.py(직접 구현한 cv2 파이프라인, 모션블러→다운스케일→센서노이즈→
JPEG 4단계)를 대체하는 것이 아니라 나란히 둔다. degrade.py 는 real_val_lq(§1차)
평가셋에 계속 쓰이고, 이 모듈은 3.1/3.2 방법론이 명시한 "Albumentations를
활용한 블러링·다운스케일링" 학습 증강을 담당한다. 두 구현이 물리적으로 같은
현상(모션블러, 해상도 저하)을 모사하므로 결과가 크게 다르지는 않지만, 실제로
쓰는 라이브러리를 방법론 서술과 일치시키기 위해 분리했다.
"""

from __future__ import annotations

import albumentations as A
import numpy as np

# NO_ALBUMENTATIONS_UPDATE=1 을 코드에서 강제하고 싶지만 환경변수는 프로세스
# 시작 전에 설정돼야 하므로, 이 값은 스크립트 상단에서 os.environ 으로 잡는다.

RANGES = {
    "motion_blur_limit": (3, 15),   # px, degrade.py 와 동일 범위
    "downscale_range": (0.18, 0.45),  # 원본 대비 축소 비율
}


def build_pipeline(rng_seed: int | None = None) -> A.Compose:
    return A.Compose([
        A.MotionBlur(blur_limit=RANGES["motion_blur_limit"], p=1.0),
        A.Downscale(scale_min=RANGES["downscale_range"][0],
                   scale_max=RANGES["downscale_range"][1],
                   interpolation=1, p=1.0),
    ])


def degrade_bgr(bgr: np.ndarray, pipeline: A.Compose) -> np.ndarray:
    """cv2 BGR 배열을 받아 그대로 BGR로 반환 (Albumentations는 채널 순서를 신경 안 씀 —
    각 변환이 순수 배열 연산이라 RGB/BGR 어느 쪽이든 동일하게 적용된다)."""
    return pipeline(image=bgr)["image"]
