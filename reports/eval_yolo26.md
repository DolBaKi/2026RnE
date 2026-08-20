# YOLO26 — baseline vs 파인튜닝 (진짜 저화질 실사진 평가)

| 세트 | 지표 | baseline | finetuned | Δ |
|---|---|---|---|---|
| real_lowq (bear) | map50 | 0.893 | 0.726 | -0.167 |
| real_lowq (bear) | recall | 0.811 | 0.641 | -0.170 |
| real_lowq (bear) | precision | 0.912 | 0.846 | -0.066 |
| real_lowq_restored (bear) | map50 | 0.222 | 0.162 | -0.060 |
| real_lowq_restored (bear) | recall | 0.206 | 0.154 | -0.051 |
| real_lowq_restored (bear) | precision | 0.829 | 0.696 | -0.133 |
| real_val (bear) | map50 | 0.908 | 0.753 | -0.154 |
| real_val (bear) | recall | 0.826 | 0.720 | -0.106 |
| real_val (bear) | precision | 0.842 | 0.911 | +0.069 |
| reg_holdout (79클래스) | map50 | 0.730 | 0.471 | -0.259 |
| reg_holdout (79클래스) | map5095 | 0.570 | 0.319 | -0.251 |