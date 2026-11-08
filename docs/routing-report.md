# Routing evaluation

RouteLLM `gpt4_dataset` (Apache-2.0): 10,000 real prompts with a GPT-4 answer, a Mixtral 8x7B answer and a GPT-4 judge's 1-5 score of the Mixtral answer. A score of 4 or more counts as good enough; GPT-4's answer counts as always good enough. Trained on 7,000, threshold chosen on 1,000, **every number below measured on the 2,000 held-out prompts**.

On its own the small model is good enough for **85.9%** of prompts. The router's job is to find the rest.

## At the chosen threshold

Target: keep quality at 95% of always using the large model. Threshold 0.878 on the router's probability that the small model will do.

| | Share to small model | Quality | Cost | Saving vs all-large |
|---|---|---|---|---|
| Always large | 0% | 100.0% | $16.64 | |
| **Router** | **56.0%** | **96.5%** | **$6.51** | **60.9%** |
| Always small | 100% | 85.9% | $0.46 | 97.2% |

Prices assumed (USD per million tokens): small 0.60 in / 0.60 out, large 10.00 in / 30.00 out; tokens estimated as characters / 4 from the actual prompts and answers.

## Against simpler routers

Router test AUC 0.738. APGR is the share of the quality gap between all-small and all-large that a router recovers, averaged over every split of traffic (RouteLLM's metric; random is 0.5).

| Router | APGR | Quality with 50% to small | Quality with 80% to small |
|---|---|---|---|
| Embedding + shape features (shipped) | 0.703 | 97.3% | 91.9% |
| Shape features only | 0.648 | 96.4% | 91.0% |
| Shortest prompts to small | 0.646 | 96.4% | 90.9% |
| Random | 0.459 | 92.3% | 88.1% |

## Quality and cost across the whole range (test set)

| Share to small | Quality | Cost |
|---|---|---|
| 0% | 100.0% | $16.64 |
| 10% | 99.8% | $14.78 |
| 20% | 99.3% | $12.96 |
| 30% | 98.9% | $11.27 |
| 40% | 98.3% | $9.58 |
| 50% | 97.3% | $7.71 |
| 60% | 95.8% | $5.76 |
| 70% | 94.1% | $3.83 |
| 80% | 91.9% | $2.48 |
| 90% | 89.3% | $1.34 |
| 100% | 85.9% | $0.46 |

## By source of prompt (test set, sources with 40+ prompts)

| Source | Prompts | Small model good enough | Routed to small | Quality |
|---|---|---|---|---|
| lmsys-chat-1m | 611 | 86% | 52% | 97.1% |
| sharegpt | 404 | 89% | 62% | 96.0% |
| hh-rlhf | 323 | 97% | 95% | 96.9% |
| flan_v2_niv2 | 223 | 68% | 2% | 99.6% |
| evol_instruct | 162 | 83% | 39% | 96.9% |
| ultrachat | 149 | 90% | 74% | 94.0% |
| flan_v2_cot | 44 | 80% | 18% | 100.0% |
| false_qa | 41 | 88% | 95% | 87.8% |

## Limits

- The judge is a model, not a person, and the labels are for one pair of models. Retrain on your own traffic and your own models' graded answers before trusting the threshold.
- Quality here is "good enough by the judge", not task success. The proxy's cost per completed task metric is the production check.
