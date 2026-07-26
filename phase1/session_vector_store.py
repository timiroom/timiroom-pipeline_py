import threading

from common.document_chunk import DocumentChunk


class SessionVectorStore:
    def __init__(self):
        self._store: dict[str, list[DocumentChunk]] = {}
        self._lock = threading.Lock()

    def put(self, session_id: str, chunks: list[DocumentChunk]) -> None:
        with self._lock:
            existing = self._store.get(session_id)
            self._store[session_id] = existing + chunks if existing else chunks

    def get(self, session_id: str) -> list[DocumentChunk]:
        with self._lock:
            return self._store.get(session_id, [])

    def has_session(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._store

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._store.pop(session_id, None)
