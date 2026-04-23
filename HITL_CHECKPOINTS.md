# HITL and Checkpoints

## Checkpoint Stages

- `post-load_pre-chunk`
- `post-chunk_pre-index`
- `post-retrieval_pre-generation`
- `post-generation_pre-final-answer`
- `pre-persist-destructive-admin-action`
- `pre-index-rebuild`

## Human-in-the-Loop Controls

- Approve or reject prepared chunk sets before committing them
- Approve or reject context review before answer generation
- Approve or reject final answer publication when enabled
- Approve or reject destructive docstore actions
- Approve or reject index rebuild requests

## Guardrails

Guardrails monitor:

- answer confidence
- citation coverage
- retrieval score floor
- Self-RAG reflection outcome

If a threshold fails, the answer is marked as `needs review` in the Chat page.
