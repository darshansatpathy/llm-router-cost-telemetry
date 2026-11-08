# 1. Route with a small model trained on graded traffic, at a threshold chosen for a quality target

Status: accepted

## Context

Rules like "short prompts go to the small model" are easy to write and wrong in both directions: short prompts can be hard ("prove that...") and long ones easy (a document to summarise). What matters is whether the small model's answer would be good enough, which only graded answers can tell.

## Decision

- Train a logistic regression on the prompt's MiniLM embedding plus six shape features (length, lines, code, maths, reasoning words, questions) to predict that the small model's answer will be judged good enough (4 or 5 out of 5).
- Train on RouteLLM's public `gpt4_dataset` (real prompts, GPT-4 and Mixtral answers, GPT-4 judge scores); hold out 10% to choose the threshold for a quality target and 20% to measure everything reported.
- Ship the weights and threshold as plain JSON (`models/router.json`), never a pickle, so loading a router cannot run code.
- The threshold is configuration: it can be overridden per deployment without retraining.

## Consequences

The judge's labels are for one pair of models. A deployment with different models should regrade a sample of its own traffic and retrain; the training command and report are built for that.
