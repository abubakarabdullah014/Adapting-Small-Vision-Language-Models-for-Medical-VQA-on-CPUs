"""Medical VQA data loaders for VQA-RAD, SLAKE-EN, and PathVQA."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from datasets import Dataset, load_dataset
from torch.utils.data import Dataset as TorchDataset


CLOSED_YES_NO = {"yes", "no", "y", "n"}
CLOSED_PATTERN = re.compile(
    r"^(yes|no|y|n|\d+|left|right|male|female|man|woman)$",
    re.IGNORECASE,
)


@dataclass
class MedVQASample:
    """One medical VQA example."""

    sample_id: str
    image: Any
    question: str
    answer: str
    answer_type: str  # "closed" | "open"
    dataset: str
    split: str
    meta: Optional[Dict[str, Any]] = None


def classify_answer_type(answer: str, hint: Optional[str] = None) -> str:
    """Classify a question as closed or open from the answer / hint."""
    if hint is not None:
        h = hint.strip().lower()
        if h in {"closed", "yes/no", "yesno", "choose"}:
            return "closed"
        if h in {"open", "free-form", "freeform"}:
            return "open"
    a = (answer or "").strip().lower()
    if a in CLOSED_YES_NO or CLOSED_PATTERN.match(a):
        return "closed"
    # Short categorical answers (<= 3 tokens, no long free text)
    tokens = a.split()
    if len(tokens) <= 2 and len(a) <= 20:
        return "closed"
    return "open"


def build_prompt(question: str, answer_type: str = "open", family: str = "generic") -> str:
    """Instruction-style prompt used at train and eval time.

    For SmolVLM / Idefics-style models, include an explicit <image> token.
    """
    if answer_type == "closed":
        body = (
            "You are a medical visual question answering assistant. "
            "Answer with a short closed-form answer (e.g., yes/no or a short label).\n"
            f"Question: {question}\nAnswer:"
        )
    else:
        body = (
            "You are a medical visual question answering assistant. "
            "Answer briefly and accurately.\n"
            f"Question: {question}\nAnswer:"
        )
    if family in {"smolvlm", "idefics", "idefics3"}:
        return f"<image>\n{body}"
    return body


def _normalize_answer(ans: str) -> str:
    return re.sub(r"\s+", " ", (ans or "").strip().lower())


def _load_vqa_rad(
    split: str,
    cache_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> List[MedVQASample]:
    """Load VQA-RAD from Hugging Face (flaviagiammarino/vqa-rad)."""
    # Prefer the public HF mirror; fall back to a local synthetic stub if offline.
    try:
        ds = load_dataset(
            "flaviagiammarino/vqa-rad",
            split=split if split != "val" else "test",
            cache_dir=cache_dir,
        )
    except Exception:
        # Alternate known mirror
        try:
            ds = load_dataset("VizWiz/vqa-rad", split=split, cache_dir=cache_dir)
        except Exception as exc:
            raise RuntimeError(
                "Could not download VQA-RAD. Ensure network access or place data under data_cache/."
            ) from exc

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    samples: List[MedVQASample] = []
    for i, row in enumerate(ds):
        answer = str(row.get("answer", row.get("answers", "")))
        if isinstance(row.get("answers"), list) and row["answers"]:
            answer = str(row["answers"][0])
        q = str(row.get("question", ""))
        hint = row.get("answer_type") or row.get("question_type")
        atype = classify_answer_type(answer, hint=str(hint) if hint else None)
        img = row["image"]
        # Materialize a detached RGB copy so dataset file handles can close
        if hasattr(img, "convert"):
            img = img.convert("RGB").copy()
        samples.append(
            MedVQASample(
                sample_id=f"vqarad_{split}_{i}",
                image=img,
                question=q,
                answer=_normalize_answer(answer),
                answer_type=atype,
                dataset="vqa_rad",
                split=split,
                meta={"raw_answer_type": hint},
            )
        )
    return samples


def _load_slake(
    split: str,
    cache_dir: Optional[str] = None,
    lang: str = "en",
    max_samples: Optional[int] = None,
) -> List[MedVQASample]:
    """Load SLAKE English split from a working HF mirror."""
    last_err: Optional[Exception] = None
    ds = None
    split_name = "validation" if split == "val" else split
    # English-ready mirrors that currently exist on the Hub
    for repo in (
        "mdwiratathya/SLAKE-vqa-english",
        "BoKelvin/SLAKE",
        "jspetrisko/slake-simplified",
    ):
        try:
            ds = load_dataset(repo, split=split_name, cache_dir=cache_dir)
            break
        except Exception as e:
            last_err = e
            ds = None
            continue
    if ds is None:
        raise RuntimeError(f"Could not download SLAKE: {last_err}")

    # Select a small slice BEFORE decoding images (avoids RAM / time blow-ups)
    n = len(ds)
    take = n if max_samples is None else min(max_samples, n)
    ds = ds.select(range(take))

    samples: List[MedVQASample] = []
    for i, row in enumerate(ds):
        row_lang = str(row.get("q_lang", row.get("lang", row.get("language", "en")))).lower()
        if row_lang in {"zh", "chinese", "cn", "ja", "jp"}:
            continue
        answer = str(row.get("answer", row.get("answers", "")))
        if isinstance(row.get("answers"), list) and row["answers"]:
            answer = str(row["answers"][0])
        q = str(row.get("question", row.get("query", "")))
        if not q:
            continue
        hint = row.get("answer_type") or row.get("content_type")
        atype = classify_answer_type(answer, hint=str(hint) if hint else None)
        img = row.get("image") or row.get("img") or row.get("images")
        if img is None:
            continue
        if hasattr(img, "convert"):
            img = img.convert("RGB").copy()
        samples.append(
            MedVQASample(
                sample_id=f"slake_{split}_{i}",
                image=img,
                question=q,
                answer=_normalize_answer(answer),
                answer_type=atype,
                dataset="slake",
                split=split,
                meta={"raw_answer_type": hint},
            )
        )
    if not samples:
        raise RuntimeError("SLAKE loaded but produced 0 samples after filtering.")
    return samples


def _load_pathvqa(split: str, cache_dir: Optional[str] = None, max_samples: Optional[int] = 500) -> List[MedVQASample]:
    """Load PathVQA; default to a test subset for cross-domain checks."""
    last_err: Optional[Exception] = None
    ds = None
    for repo in ("flaviagiammarino/path-vqa", "flaviagiammarino/pathvqa", "PathVQA/PathVQA"):
        try:
            ds = load_dataset(repo, split=split if split != "val" else "test", cache_dir=cache_dir)
            break
        except Exception as e:
            last_err = e
            continue
    if ds is None:
        raise RuntimeError(f"Could not download PathVQA: {last_err}")

    samples: List[MedVQASample] = []
    for i, row in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break
        answer = str(row.get("answer", ""))
        q = str(row.get("question", ""))
        hint = row.get("answer_type")
        atype = classify_answer_type(answer, hint=str(hint) if hint else None)
        samples.append(
            MedVQASample(
                sample_id=f"pathvqa_{split}_{i}",
                image=row["image"],
                question=q,
                answer=_normalize_answer(answer),
                answer_type=atype,
                dataset="pathvqa",
                split=split,
                meta={"raw_answer_type": hint},
            )
        )
    return samples


def load_medvqa(
    name: str,
    split: str = "train",
    cache_dir: Optional[str] = "data_cache",
    max_samples: Optional[int] = None,
) -> List[MedVQASample]:
    """
    Load a medical VQA split.

    Args:
        name: one of vqa_rad | slake | pathvqa
        split: train | val | test
        cache_dir: HF datasets cache
        max_samples: optional hard cap (applied after load)
    """
    name = name.lower().replace("-", "_")
    if name in {"vqa_rad", "vqarad"}:
        samples = _load_vqa_rad(split, cache_dir=cache_dir, max_samples=max_samples)
    elif name in {"slake", "slake_en"}:
        samples = _load_slake(split, cache_dir=cache_dir, max_samples=max_samples)
    elif name in {"pathvqa", "path_vqa"}:
        # PathVQA uses max_samples inside loader by default for test
        samples = _load_pathvqa(split, cache_dir=cache_dir, max_samples=max_samples or 500)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return samples


def to_hf_dataset(samples: Sequence[MedVQASample]) -> Dataset:
    """Convert sample list to HuggingFace Dataset (images kept as PIL)."""
    rows = {
        "sample_id": [s.sample_id for s in samples],
        "image": [s.image for s in samples],
        "question": [s.question for s in samples],
        "answer": [s.answer for s in samples],
        "answer_type": [s.answer_type for s in samples],
        "dataset": [s.dataset for s in samples],
        "split": [s.split for s in samples],
        "prompt": [build_prompt(s.question, s.answer_type) for s in samples],
    }
    return Dataset.from_dict(rows)


class MedVQADataset(TorchDataset):
    """Torch Dataset wrapping MedVQASample list with optional processor."""

    def __init__(
        self,
        samples: Sequence[MedVQASample],
        processor=None,
        max_length: int = 512,
        for_training: bool = True,
    ):
        self.samples = list(samples)
        self.processor = processor
        self.max_length = max_length
        self.for_training = for_training

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        prompt = build_prompt(s.question, s.answer_type)
        item: Dict[str, Any] = {
            "sample_id": s.sample_id,
            "image": s.image,
            "question": s.question,
            "answer": s.answer,
            "answer_type": s.answer_type,
            "prompt": prompt,
            "dataset": s.dataset,
        }
        if self.processor is None:
            return item

        # Generic VLM processor path: text + image; labels = answer tokens when training
        text = prompt + (f" {s.answer}" if self.for_training else "")
        try:
            enc = self.processor(
                text=[text],
                images=[s.image],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )
        except TypeError:
            # Some processors use different kwargs
            enc = self.processor(text=text, images=s.image, return_tensors="pt")

        out = {k: v.squeeze(0) for k, v in enc.items()}
        if self.for_training and "input_ids" in out:
            labels = out["input_ids"].clone()
            # Mask pad tokens if present
            if "attention_mask" in out:
                labels = labels.masked_fill(out["attention_mask"] == 0, -100)
            out["labels"] = labels
        out.update({k: item[k] for k in ("sample_id", "answer", "answer_type", "question", "prompt")})
        return out


__all__ = [
    "MedVQASample",
    "MedVQADataset",
    "load_medvqa",
    "build_prompt",
    "classify_answer_type",
    "to_hf_dataset",
]
