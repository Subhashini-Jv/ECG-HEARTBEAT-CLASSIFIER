| variant | acc | macro recall | N | S | V | F |
|---|---|---|---|---|---|---|
| single model, default recipe | 0.904 | 0.620 | 0.925 | 0.661 | 0.852 | 0.041 |
| + raw beat channel (not patient-normalised) | 0.911 | 0.644 | 0.928 | 0.719 | 0.886 | 0.041 |
| no RR-ratio features | 0.107 | 0.248 | 0.089 | 0.695 | 0.000 | 0.206 |
| unweighted loss | 0.941 | 0.464 | 0.996 | 0.056 | 0.801 | 0.003 |
| intra-patient split (leaky, for contrast) | 0.921 | 0.914 | 0.919 | 0.925 | 0.943 | 0.869 |
