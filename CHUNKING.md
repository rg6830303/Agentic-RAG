# Chunking

## Implemented Methods

- `fixed`: stable character-window chunking with overlap
- `semantic`: sentence-oriented grouping for dense prose
- `recursive`: delimiter-aware splitting for code and structured text
- `adaptive`: heuristic router that internally chooses a better base strategy
- `hierarchical`: parent/child chunk graph for coherent hierarchical retrieval
- `auto`: per-file strategy selection based on extension, headings, line density, and delimiter profile

## Auto Heuristics

Auto mode considers:

- file extension and code-like patterns
- heading density
- line density
- blank-line and bullet structure
- delimiter patterns such as commas, braces, and colons
- overall document length

## HITL Support

The Ingestion page previews chunk plans before persistence. When chunk approval is enabled, no chunks are committed until the user explicitly approves the prepared chunk set.
