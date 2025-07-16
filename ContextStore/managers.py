from collections import deque
from typing import TypeVar, Generic, Callable, Dict, Optional, List, Union, Protocol

from Agents.ManagerAgent.DB import sql_client
from langgraph.graph.message import add_messages  # if still needed
from langchain_google_genai import ChatGoogleGenerativeAI  # if still needed
from .pointers import ImagePointer,DocumentPointer
from collections import deque

T = TypeVar("T", bound=Union[ImagePointer, DocumentPointer])


class PointerManager(Generic[T]):
    """
    Generic manager for any "pointer" type T that implements HasID:
      - get_id() -> str
      - persistence handled via callbacks
    """
    def __init__(
        self,
        *,
        maxlen: int,
        add_to_db: Callable[[T], None],
        delete_from_db: Callable[[T], None]
    ):
        self._store: Dict[str, T] = {}
        self._window: deque[T] = deque(maxlen=maxlen)
        self._add_to_db = add_to_db
        self._delete_from_db = delete_from_db

    def add(self, ptr: T, to_front: bool = False) -> None:
        """Add or update pointer in store + window (move-to-front support)."""
        pid = ptr.get_id()
        # Persist to DB
        # self._add_to_db(ptr)
        # Update in-memory store
        self._store[pid] = ptr

        # Evict existing from window
        self._window = deque(
            (p for p in self._window if p.get_id() != pid),
            maxlen=self._window.maxlen
        )
        # Insert new or reinsertion
        if to_front:
            self._window.appendleft(ptr)
        else:
            self._window.append(ptr)

    def get(self, pid: str) -> Optional[T]:
        return self._store.get(pid)

    def remove(self, pid: str, delete_from_db: bool = True) -> bool:
        ptr = self._store.pop(pid, None)
        if not ptr:
            return False
        if delete_from_db:
            self._delete_from_db(ptr)
        # Remove from window
        self._window = deque(
            (p for p in self._window if p.get_id() != pid),
            maxlen=self._window.maxlen
        )
        return True

    def recent(self) -> List[T]:
        """Return window in MRU → LRU order (leftmost is most recent)."""
        return list(self._window)

    def all_ids(self) -> List[str]:
        return list(self._store.keys())

    def all(self) -> List[T]:
        return list(self._store.values())
