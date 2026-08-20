# baseline vs 파인튜닝 v1(불균형) vs v2(정칙화 오버샘플링) 평가 결과

| 세트 | 지표 | baseline | v1 | v2 | v2-v1 |
|---|---|---|---|---|---|
| real_val (bear) | map50 | 0.904 | 0.501 | 0.448 | -0.053 |
| real_val (bear) | recall | 0.831 | 0.310 | 0.304 | -0.006 |
| real_val (bear) | precision | 0.873 | 0.853 | 0.643 | -0.210 |
| real_val_lq (bear) | map50 | 0.399 | 0.478 | 0.497 | +0.019 |
| real_val_lq (bear) | recall | 0.282 | 0.451 | 0.451 | +0.000 |
| real_val_lq (bear) | precision | 0.660 | 0.619 | 0.677 | +0.058 |
| reg_holdout (79클래스) | map50 | 0.728 | 0.474 | 0.494 | +0.020 |
| reg_holdout (79클래스) | map5095 | 0.571 | 0.302 | 0.331 | +0.029 |