"""8차 — v2 모델(성별 맞춤·CCTV 화면화 이전 합성으로 학습)을 v3 테스트(CCTV 화면화)로 재평가.

v2 와 v3 는 인파 테스트 프레임이 달라 표의 숫자를 직접 비교할 수 없다. 같은 v3 테스트에서
두 버전의 학습 데이터를 비교해, 성별 맞춤 + CCTV 화면 생성형 스타일화가 학습 데이터로서
나아졌는지를 분리한다. 실사진 평가 세트는 두 버전에서 동일하다.

입력  runs/face10/v2_crowdneg/*.pt   출력  reports/eval_face10_cross_v2model_v3test.json
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rareobj import paths  # noqa: E402

t43 = importlib.import_module("43_train_eval_face10")


def main() -> int:
    split, rows, *_ = t43.load_items()
    out = {}
    for ck in sorted((t43.RUNS / "v2_crowdneg").glob("*.pt")):
        arm, seed = ck.stem.rsplit("_s", 1)
        key = f"{arm.replace('real_synth', 'real+synth')}/s{seed}"
        m = t43.t34.load_backbone().cuda().eval()
        m.load_state_dict(torch.load(ck))
        out[key] = t43.evaluate(m, split, rows)
        print(key, round(out[key]["crowd"]["find_rate@far1%"], 3), round(out[key]["crowd"]["top1_detected"], 3), flush=True)
    (paths.REPORTS / "eval_face10_cross_v2model_v3test.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
