from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Type

from pydantic import BaseModel


@dataclass(frozen=True)
class LLMPreset:
    name: str
    model: Optional[str] = None
    temperature: Optional[float] = None
    response_schema: Optional[Type[BaseModel]] = None
    json_mode: bool = False
