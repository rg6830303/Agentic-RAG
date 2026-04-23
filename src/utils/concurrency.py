from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def parallel_map(
    items: Iterable[T],
    worker: Callable[[T], R],
    max_workers: int = 4,
    enabled: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[R]:
    materialized = list(items)
    total = len(materialized)
    if total == 0:
        return []
    if not enabled or max_workers <= 1 or total == 1:
        results: list[R] = []
        for index, item in enumerate(materialized, start=1):
            results.append(worker(item))
            if progress_callback:
                progress_callback(index, total)
        return results

    results: list[R | None] = [None] * total
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(worker, item): index for index, item in enumerate(materialized)
        }
        completed = 0
        for future in as_completed(future_map):
            index = future_map[future]
            results[index] = future.result()
            completed += 1
            if progress_callback:
                progress_callback(completed, total)
    return [result for result in results if result is not None]
