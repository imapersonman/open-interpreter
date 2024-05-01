from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator, Generic, List, Optional, TypeVar

import litellm


@dataclass
class BaseLlm(ABC):
    model: str
    temperature: float = 0
    supports_vision: bool = False

    context_window: Optional[int] = None
    max_tokens: Optional[int] = None
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    api_version: Optional[str] = None
    max_budget: Optional[int] = None

    def __post_init__(self):
        self.support_functions = (
            self.model in litellm.model_cost
            and litellm.supports_function_calling(model=self.model)
        )

    @abstractmethod
    def run(self, messages: List) -> Generator: ...
