# Evaluation

## Local-First Modes

- `HeuristicEvaluator`: always available, stores JSON, CSV, and Markdown reports under `artifacts/evaluations/`
- `RagasLiteLLMEvaluator`: optional adapter surface that only activates if `ragas`, `litellm`, and Azure chat configuration are all present

## Golden Dataset

The repository includes an original NCERT Class 12 Physics-inspired benchmark in `data/golden_eval/ncert_physics_golden.json` and a matching sample corpus under `data/sample_corpus/ncert_physics/`.

## Report Outputs

Each evaluation run creates a dedicated report directory with:

- `report.json`
- `report.csv`
- `report.md`
