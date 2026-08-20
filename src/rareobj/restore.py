"""NAFNet 디블러링 → Real-ESRGAN 업스케일링 복원 파이프라인.

순서를 디블러 먼저, 업스케일 나중으로 둔 이유: 저화질 입력은 이미 다운스케일
왕복으로 디테일이 뭉개진 상태라, 원 해상도에서 모션블러부터 먼저 제거하는 쪽이
Real-ESRGAN이 없는 디테일을 "환각"으로 채워 넣기 전에 실제 신호를 최대한
복원해 두는 순서다. 업스케일을 먼저 하면 블러가 4배 커진 캔버스에 그대로
남아 NAFNet이 더 넓은 커널로 지워야 하는 반대 상황이 된다.

최종적으로 원본 크기로 되돌리는 이유: Real-ESRGAN은 4배 업스케일 전용으로
학습돼 있어 임의 배율을 직접 내지 못한다. 4배로 올린 뒤 원래 크기로 다운스케일
하면, 그 축소 과정 자체가 초해상도 모델이 복원한 고주파 디테일을 앨리어싱 없이
목표 해상도로 응축하는 저역통과 필터 역할을 한다 — 단순 디노이즈보다 강한
효과를 낸다.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .vendor.nafnet_arch import GOPRO_WIDTH32_CONFIG, NAFNetLocal
from .vendor.rrdbnet_arch import X4PLUS_CONFIG, RRDBNet

NAFNET_CKPT_URL = "https://huggingface.co/nyanko7/nafnet-models/resolve/main/NAFNet-GoPro-width32.pth"
ESRGAN_CKPT_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"


def load_nafnet(ckpt_path: Path, device: str = "cuda") -> NAFNetLocal:
    # NAFNetLocal(TLC 적용판)을 쓴다 — 이유는 vendor/nafnet_arch.py 의 NAFNetLocal
    # docstring 참고. 학습 크롭 256x256 기준은 공식 GoPro 설정값 그대로.
    model = NAFNetLocal(**GOPRO_WIDTH32_CONFIG, train_size=(1, 3, 256, 256))
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd["params"], strict=True)
    return model.to(device).eval()


def load_esrgan(ckpt_path: Path, device: str = "cuda") -> RRDBNet:
    model = RRDBNet(**X4PLUS_CONFIG)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd["params_ema"], strict=True)
    return model.to(device).eval()


def _to_tensor(bgr: np.ndarray, device: str) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)


def _to_image(t: torch.Tensor) -> np.ndarray:
    rgb = t.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return cv2.cvtColor((rgb * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)


class Restorer:
    """NAFNet(디블러) → Real-ESRGAN(4x 업스케일) → 원본 크기로 다운스케일."""

    def __init__(self, nafnet_ckpt: Path, esrgan_ckpt: Path, device: str = "cuda"):
        self.device = device
        self.nafnet = load_nafnet(nafnet_ckpt, device)
        self.esrgan = load_esrgan(esrgan_ckpt, device)

        """실제로 겪은 두 가지 불안정 사례와 각각의 처리.

        (1) 균일한 밝은 영역이 넓은 사진에서 NAFNet 출력이 [0,1]을 크게 벗어나
        폭주했다(-27~36, 심한 경우 -158~390). 우선 plain NAFNet의 SCA가 쓰는
        전역 평균풀링이 학습 크롭(256x256)과 테스트 해상도(수백~수천 px) 사이의
        통계 불일치를 일으킨다고 보고 원 저자들의 TLC(NAFNetLocal, 학습 크롭
        기준 국소평균으로 대체)를 적용했다 — 실제로 폭주 폭이 줄었지만
        완전히는 없어지지 않았다. 두 모델을 이어붙이면 한쪽의 폭주가 다음
        모델로 그대로 전파되므로 NAFNet과 ESRGAN 사이에 clamp(0,1)을 둔다.

        (2) 그런데 clamp를 넣고도 특정 입력(흐릿하고 디테일이 적은 사진 —
        공교롭게도 real_lowq가 고른 사진들의 특성 그 자체)에서 Real-ESRGAN이
        규칙적인 격자무늬를 만들어냈다. 원인을 추적해 보니 clamp된 정상 범위의
        입력을 ESRGAN에 넣어도 재현됐다 — 즉 NAFNet의 수치 폭주 문제가 아니라
        **Real-ESRGAN 자체가 매끈하고 정보가 적은 영역에서 고주파 디테일을
        "환각"으로 채워 넣다가 규칙적인 격자 아티팩트를 만드는, GAN 기반
        초해상도 모델의 잘 알려진 실패 양상**이다. 이건 코드로 고칠 수 있는
        버그가 아니라 이 모델의 한계이므로 고치지 않고 결과에 그대로 반영했다
        — docs/04_yolo26_pipeline.md 에 그 영향(복원 전처리가 오히려 검출
        성능을 크게 떨어뜨림)을 정직하게 기록해 뒀다.
        """
        h, w = bgr.shape[:2]
        x = _to_tensor(bgr, self.device)
        x = self.nafnet(x).clamp(0, 1)
        x = self.esrgan(x)
        up = _to_image(x)  # h*4, w*4
        return cv2.resize(up, (w, h), interpolation=cv2.INTER_AREA)
