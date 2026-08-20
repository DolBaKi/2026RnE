# baseline vs 파인튜닝 평가 결과

| 세트 | 지표 | baseline | finetuned | Δ |
|---|---|---|---|---|
| real_val (bear) | map50 | 0.904 | 0.501 | -0.402 |
| real_val (bear) | recall | 0.831 | 0.310 | -0.521 |
| real_val (bear) | precision | 0.873 | 0.853 | -0.021 |
| real_val_lq (bear) | map50 | 0.399 | 0.478 | +0.079 |
| real_val_lq (bear) | recall | 0.282 | 0.451 | +0.169 |
| real_val_lq (bear) | precision | 0.660 | 0.619 | -0.040 |
| reg_holdout (79클래스 전체) | map50 | 0.728 | 0.474 | -0.254 |
| reg_holdout (79클래스 전체) | map5095 | 0.571 | 0.302 | -0.269 |