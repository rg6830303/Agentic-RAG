# Python Programming Fundamentals

Python is a high-level, interpreted, dynamically typed, general-purpose programming language created by Guido van Rossum and first released in 1991. The current major version is Python 3. The Python Software Foundation manages the language reference implementation, CPython.

## Core data types
- **Numeric**: `int` (arbitrary precision), `float`, `complex`, `bool`.
- **Sequence**: `list` (mutable), `tuple` (immutable), `range`, `str`, `bytes`, `bytearray`.
- **Mapping**: `dict` (insertion-ordered since Python 3.7).
- **Set**: `set` (mutable, unordered, unique), `frozenset` (immutable).
- **None**: the `None` singleton represents the absence of a value.

## Control flow
- `if`/`elif`/`else`, `for`, `while`, `break`, `continue`.
- `match`/`case` structural pattern matching (Python 3.10+).
- Comprehensions: list, set, dict, and generator expressions.

## Functions and objects
- Functions are first-class: can be passed around, returned, and stored.
- Default arguments are evaluated **once** at function definition time — a common pitfall when mutable defaults like lists or dicts are used.
- Classes use `class` keyword; everything inherits from `object`. Dataclasses (`@dataclass`) reduce boilerplate for plain data containers.
- Type hints (`def f(x: int) -> str:`) are optional at runtime but enable static analysis with `mypy`, `pyright`, and IDE tooling.

## Modules and packages
- A module is any `.py` file. A package is a directory containing `__init__.py` (implicit namespace packages since 3.3).
- Standard library highlights: `os`, `sys`, `pathlib`, `json`, `re`, `datetime`, `collections`, `itertools`, `functools`, `typing`, `dataclasses`, `asyncio`, `concurrent.futures`, `subprocess`, `logging`, `argparse`, `unittest`, `sqlite3`.

## Async programming
- Coroutines are defined with `async def` and awaited with `await`.
- `asyncio` is the standard event loop implementation.
- For HTTP, libraries like `httpx`, `aiohttp` provide async clients.

## Popular third-party libraries
- Web: FastAPI, Flask, Django, Starlette.
- Data: NumPy, pandas, Polars, DuckDB.
- ML: PyTorch, TensorFlow, JAX, scikit-learn, transformers (Hugging Face).
- Vector search: FAISS, chromadb, pgvector (via psycopg).
- Testing: pytest.
- Packaging: pip, uv, poetry, hatch.
