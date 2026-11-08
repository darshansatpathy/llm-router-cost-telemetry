"""The routing dataset: RouteLLM's `gpt4_dataset` (Apache-2.0).

10,000 real prompts (from LMSYS-Chat, ShareGPT, HH-RLHF, FLAN and others), each with an answer from
a strong model (GPT-4) and from a smaller one (Mixtral 8x7B), and a GPT-4 judge's 1-5 score of the smaller
model's answer. We treat a score of 4 or 5 as "the small model was good enough", and the strong model's
answer as always acceptable. The data is downloaded, not redistributed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

URL = "https://huggingface.co/datasets/routellm/gpt4_dataset/resolve/main/valid.jsonl"
ACCEPT = 4


@dataclass(frozen=True)
class Example:
    prompt: str
    source: str
    small_score: int
    small_answer: str
    large_answer: str

    @property
    def small_ok(self) -> bool:
        return self.small_score >= ACCEPT


def tokens(text: str) -> int:
    """About four characters per token for English; used only to price the replayed answers."""
    return max(1, len(text) // 4)


def download(dest: Path) -> Path:
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        with httpx.stream("GET", URL, follow_redirects=True, timeout=300) as r:
            r.raise_for_status()
            with dest.open("wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
    return dest


def source_name(raw: str) -> str:
    """The prompt's source, by its public dataset name (the helpful-harmless set is shown as hh-rlhf)."""
    return "hh-rlhf" if raw.endswith("-hh") else raw


def load(path: Path) -> list[Example]:
    out = []
    for line in path.read_text().splitlines():
        d = json.loads(line)
        if d.get("mixtral_score") is None or not d.get("prompt"):
            continue
        out.append(
            Example(
                d["prompt"],
                source_name((d.get("source") or ["?"])[0]),
                int(d["mixtral_score"]),
                d["mixtral_response"] or "",
                d["gpt4_response"] or "",
            )
        )
    return out
