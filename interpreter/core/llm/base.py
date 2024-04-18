from typing import Generic, TypeVar


CompletionsStream = TypeVar("Completions")
class BaseLLM(Generic[CompletionsStream]):
    def __init__(
            self,
            settings: GlobalSettings,
            
        ):
        ...