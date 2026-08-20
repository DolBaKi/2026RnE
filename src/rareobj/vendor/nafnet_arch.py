"""NAFNet 아키텍처 — 추론에 필요한 부분만 벤더링.

원본: https://github.com/megvii-research/NAFNet
      basicsr/models/archs/{NAFNet_arch.py, arch_util.py}
Copyright (c) 2022 megvii-model. MIT License.
논문: Chen et al., "Simple Baselines for Image Restoration", ECCV 2022.

`basicsr` 패키지 전체를 설치하는 대신 이 파일 하나만 들여왔다 — basicsr은
설치 과정에서 무거운 전이 의존성(stringzilla 등, Windows에서 컴파일러 없이는
휠이 없는 경우도 있음)을 끌고 오고, 우리가 실제로 쓰는 건 순수 추론용
`NAFNet` 클래스 하나뿐이다. TLC(test-time local converter, `NAFNetLocal`)는
정확도를 살짝 더 올려주는 테스트 전용 트릭이라 여기서는 뺐다 — 없어도 동작한다.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return (
            gx,
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
            None,
        )


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, 1, 0, groups=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, 1, 0, groups=1, bias=True)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, 1, 0, groups=1, bias=True),
        )
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, 1, 0, groups=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, 1, 0, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=(), dec_blk_nums=()):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1, groups=1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3, 1, 1, groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        x = self.intro(inp)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x) + inp
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h))


# GoPro(모션블러 디블러링) width32 공식 설정.
# 체크포인트: https://huggingface.co/nyanko7/nafnet-models (NAFNet-GoPro-width32.pth)
GOPRO_WIDTH32_CONFIG = dict(img_channel=3, width=32, middle_blk_num=1,
                            enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1])


# --- TLC (Test-time Local Converter) — 원본: basicsr/models/archs/local_arch.py ---
# 실제로 겪은 문제: 평범한 NAFNet을 학습 크롭 크기(256x256)와 다른 테스트 해상도
# (예: 480x640 전체 이미지)에 그대로 돌리면, NAFBlock의 SCA가 쓰는
# nn.AdaptiveAvgPool2d(1)이 이미지 전체를 하나로 평균 내버린다 — 학습 때는 작은
# 크롭이 주는 국소 통계만 봤는데 테스트 때 전체 이미지의 전역 통계를 보게 되면서
# 분포가 완전히 달라진다. 균일한 밝은 영역(암석, 하늘 등)이 넓은 사진에서 이
# 불일치가 극단으로 튀어 NAFNet 출력이 [0,1]을 크게 벗어나며 폭주하는 것을
# 실제로 확인했다(-158~390 범위, 결과물은 체커보드 노이즈). Chu et al.
# (arXiv:2112.04491)의 TLC는 전역 평균풀링을 "학습 크롭 크기에 맞춘 이동평균"으로
# 바꿔 이 스케일 불일치를 없앤다 — NAFNetLocal이 이 대체를 자동으로 적용한다.
class AvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False, train_size=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad
        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
        self.train_size = train_size

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]
            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-1])

        if self.kernel_size[0] >= x.size(-2) and self.kernel_size[1] >= x.size(-1):
            return F.adaptive_avg_pool2d(x, 1)

        n, c, h, w = x.shape
        s = x.cumsum(dim=-1).cumsum_(dim=-2)
        s = F.pad(s, (1, 0, 1, 0))
        k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
        s1, s2, s3, s4 = s[:, :, :-k1, :-k2], s[:, :, :-k1, k2:], s[:, :, k1:, :-k2], s[:, :, k1:, k2:]
        out = (s4 + s1 - s2 - s3) / (k1 * k2)

        if self.auto_pad:
            _h, _w = out.shape[2:]
            pad2d = ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2)
            out = F.pad(out, pad2d, mode="replicate")
        return out


def _replace_avgpool(model: nn.Module, base_size, train_size) -> None:
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            _replace_avgpool(m, base_size, train_size)
        if isinstance(m, nn.AdaptiveAvgPool2d):
            setattr(model, n, AvgPool2d(base_size=base_size, train_size=train_size))


class NAFNetLocal(NAFNet):
    """테스트 시 SCA의 전역 평균풀링을 학습 크롭 크기 기준 국소 평균으로 바꾼
    NAFNet. 원본 저자들이 실제 배포용 추론에 쓰는 방식이 이것이다(plain NAFNet
    아님) — 논문 표에 나온 수치도 전부 이 버전 기준.

    더미 forward 한 번이 반드시 필요하다 — AvgPool2d.forward()는 kernel_size가
    None일 때만 (x.shape 기준으로) 계산하고, 그 뒤로는 처음 계산된 값을 그대로
    재사용한다. train_size 크기의 더미 입력으로 한 번 돌려서 kernel_size를
    "학습 크롭 기준 절대 윈도우 크기"로 고정해 둬야, 이후 실제 테스트 이미지가
    어떤 크기로 들어와도 같은 절대 윈도우로 국소 평균을 낸다(원본 TLC의 의도).
    """

    def __init__(self, *args, train_size=(1, 3, 256, 256), **kwargs):
        super().__init__(*args, **kwargs)
        base_size = (int(train_size[-2] * 1.5), int(train_size[-1] * 1.5))
        self.eval()
        with torch.no_grad():
            _replace_avgpool(self, base_size, train_size)
            self.forward(torch.rand(*train_size))
