from typing import Any, Callable, TypeVar, Optional
import tqdm

T = TypeVar("T")


class PrimaryProcessHandler:
    """Handler for operations that should only run on the primary process in distributed training"""

    def __init__(self, local_rank: int):
        self.is_primary = local_rank == 0

    def run(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> Optional[T]:
        """Execute a function only on the primary process"""
        if self.is_primary:
            return func(*args, **kwargs)
        return None

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Print only on the primary process"""
        if self.is_primary:
            print(*args, **kwargs)

    def get_progress_bar(self, iterator: Any, total: int, desc: str) -> Any:
        """Get a progress bar on the primary process, otherwise return the iterator"""
        if self.is_primary:
            return tqdm.tqdm(iterator, total=total, desc=desc)
        return iterator
