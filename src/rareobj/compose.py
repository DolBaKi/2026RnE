"""전경(생성된 곰) 분리 + 배경 합성.

작은 마스크에 직접 inpainting하면 디퓨전 모델이 그 스케일에서 의미 있는 디테일을
만들지 못한다(512 캔버스에서 32px 영역은 latent 4px 미만). 대신:

  1. 곰을 충분히 큰 스케일로 선명하게 생성 (배경은 최대한 단순하게 유도)
  2. GrabCut으로 전경/배경 분리 → RGBA 컷아웃
  3. 목표 크기로 리사이즈해 별도 생성한 배경에 합성 → bbox는 합성 좌표에서 정확히 나옴
  4. 합성본 전체에 열화 파이프라인 적용

이 순서라 자동 라벨링에 검출기 추론이 필요 없다 — bbox는 addWeighted 좌표 그 자체다.
"""

from __future__ import annotations

import cv2
import numpy as np


def grabcut_segment(bgr: np.ndarray, margin_frac: float = 0.06) -> np.ndarray | None:
    """중앙 피사체를 GrabCut으로 분리. 성공 시 0/255 마스크, 실패 시 None.

    스튜디오풍 프롬프트로 생성했으므로 피사체가 프레임 중앙을 차지한다고 가정하고
    시드 사각형을 이미지 대부분(여백만 제외)으로 잡는다.
    """
    h, w = bgr.shape[:2]
    m = int(min(h, w) * margin_frac)
    rect = (m, m, w - 2 * m, h - 2 * m)

    mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    frac = fg.mean() / 255.0
    if frac < 0.03 or frac > 0.92:
        return None  # 분할 실패로 판단 (거의 안 잡히거나 프레임 전체를 덮음)

    # 작은 구멍/노이즈 제거
    k = np.ones((5, 5), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
    if fg.sum() == 0:
        return None
    return fg


def crop_to_mask(bgr: np.ndarray, mask: np.ndarray, pad: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """마스크 bbox로 크롭한 (BGR, alpha) 쌍. alpha는 경계를 살짝 페더링한다."""
    ys, xs = np.where(mask > 0)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, mask.shape[0] - 1)
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, mask.shape[1] - 1)
    crop = bgr[y0:y1 + 1, x0:x1 + 1]
    alpha = mask[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0) / 255.0
    return crop, alpha


def resize_rgba(bgr: np.ndarray, alpha: np.ndarray, target_w: int, target_h: int) -> tuple[np.ndarray, np.ndarray]:
    interp = cv2.INTER_AREA if target_w < bgr.shape[1] else cv2.INTER_LANCZOS4
    r_bgr = cv2.resize(bgr, (target_w, target_h), interpolation=interp)
    r_a = cv2.resize(alpha, (target_w, target_h), interpolation=interp)
    return r_bgr, r_a


def match_brightness(fg_bgr: np.ndarray, alpha: np.ndarray, bg_patch: np.ndarray,
                     strength: float = 0.6) -> np.ndarray:
    """붙여넣기 자국을 줄이기 위해 전경 밝기를 배경 국소 밝기 쪽으로 완만히 당긴다."""
    fg_gray = cv2.cvtColor(fg_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg_gray = cv2.cvtColor(bg_patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
    w = alpha > 0.5
    if w.sum() < 10:
        return fg_bgr
    fg_mean = fg_gray[w].mean()
    bg_mean = bg_gray[w].mean() if w.sum() else fg_mean
    gain = 1.0 + strength * ((bg_mean - fg_mean) / 255.0)
    gain = float(np.clip(gain, 0.7, 1.35))
    return np.clip(fg_bgr.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def paste(bg: np.ndarray, fg_bgr: np.ndarray, alpha: np.ndarray, x: int, y: int) -> np.ndarray:
    """(x, y) = 좌상단. 캔버스 밖으로 나가는 부분은 잘라 붙인다."""
    out = bg.copy()
    H, W = out.shape[:2]
    h, w = fg_bgr.shape[:2]

    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return out

    fx0, fy0 = x0 - x, y0 - y
    fx1, fy1 = fx0 + (x1 - x0), fy0 + (y1 - y0)

    roi = out[y0:y1, x0:x1].astype(np.float32)
    patch = fg_bgr[fy0:fy1, fx0:fx1].astype(np.float32)
    a = alpha[fy0:fy1, fx0:fx1][..., None]
    out[y0:y1, x0:x1] = (patch * a + roi * (1 - a)).astype(np.uint8)
    return out
