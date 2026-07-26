from dataclasses import dataclass, field
from typing import Any
import uuid


@dataclass
class DocumentChunk:
    id: uuid.UUID
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    relevance_score: float | None = None
    embedding: list[float] | None = None
