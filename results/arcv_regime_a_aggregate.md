# ARC-V Regime A -- aggregate over 3 seeds [42, 1337, 2024]

Held-out = studio(Indian)-trained -> wild(SS)-test. Mean +/- std.

| Method | Held-out acc (mean +/- std) | Collapse pp (mean +/- std) | seeds |
|---|---|---|---|
| arcv | 67.32 +/- 1.72 | 32.68 +/- 1.72 | 3 |
| fourier | 66.34 +/- 0.00 | 33.66 +/- 0.00 | 1 |
| mixstyle | 64.78 +/- 0.00 | 35.22 +/- 0.00 | 1 |
| erm | 61.71 +/- 1.82 | 38.29 +/- 1.82 | 3 |

Verdict: ERM 61.71 vs ARC-V 67.32 on the held-out source. ARC-V beats ERM by +5.61 pp (margin exceeds the pooled seed spread; pooled std 2.50).