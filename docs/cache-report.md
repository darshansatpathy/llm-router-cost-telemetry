# Semantic cache calibration

PAWS test set (Zhang, Baldridge and He, 2019; data from Google LLC): 8,000 sentence pairs with heavy word overlap, 3,536 of them paraphrases and the rest different in meaning. Embedder `all-MiniLM-L6-v2`, cosine similarity.

**Chosen threshold 1.000**: the lowest whose false-hit rate on pairs that mean different things is at most 1%. At it, 0.02% of different-meaning pairs would be served the other's answer, and 0.3% of true paraphrases hit.

| Threshold | False hits on different meanings | Paraphrases that hit | Precision of hits |
|---|---|---|---|
| 0.800 | 97.98% | 98.6% | 44.4% |
| 0.820 | 97.40% | 98.4% | 44.4% |
| 0.840 | 96.66% | 97.8% | 44.5% |
| 0.860 | 95.83% | 97.2% | 44.5% |
| 0.880 | 94.47% | 96.3% | 44.7% |
| 0.900 | 91.71% | 94.9% | 45.0% |
| 0.920 | 87.93% | 92.5% | 45.4% |
| 0.940 | 81.09% | 89.6% | 46.7% |
| 0.960 | 69.58% | 82.1% | 48.3% |
| 0.980 | 45.07% | 63.7% | 52.8% |
| 0.982 | 41.51% | 60.3% | 53.5% |
| 0.984 | 37.81% | 55.9% | 53.9% |
| 0.986 | 33.80% | 51.7% | 54.8% |
| 0.988 | 29.32% | 47.1% | 56.0% |
| 0.990 | 23.63% | 42.1% | 58.5% |
| 0.992 | 17.99% | 35.6% | 61.1% |
| 0.994 | 12.14% | 27.2% | 64.0% |
| 0.996 | 6.38% | 18.4% | 69.6% |
| 0.998 | 2.04% | 8.5% | 76.8% |
| 1.000 | 0.02% | 0.3% | 91.7% |

What this means: a general-purpose sentence embedder scores "flights from Paris to Rome" and "flights from Rome to Paris" as near-identical. No threshold below exact equality keeps wrong answers rare on pairs like these, so **the semantic layer ships off**. The exact layer (the same request again) is on and safe. Turn the semantic layer on only for a tenant whose traffic is FAQ-like and where a near-miss answer is harmless, and calibrate on that tenant's own pairs first.
