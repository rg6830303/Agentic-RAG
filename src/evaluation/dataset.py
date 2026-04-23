from __future__ import annotations

import json
from pathlib import Path

from src.utils.models import EvaluationSample


def load_golden_samples(path: Path) -> list[EvaluationSample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        EvaluationSample(
            sample_id=item["sample_id"],
            question=item["question"],
            reference_answer=item["reference_answer"],
            reference_contexts=item.get("reference_contexts", []),
            expected_files=item.get("expected_files", []),
            metadata=item.get("metadata", {}),
        )
        for item in payload
    ]
