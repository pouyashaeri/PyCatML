# Text Category (optional)

from typing import Any, Callable, Dict
from .base import Category

class TextCategory(Category):
    def __init__(self):
        super().__init__()
        self.metadata_map: Dict[Any, Dict[str, Any]] = {}

    def add_text_object(self, text: Any, metadata: Dict[str, Any] = None):
        self.add_object(text)
        self.metadata_map[text] = metadata or {}

    def add_text_morphism(self, source: Any, target: Any, morphism: Callable):
        self.add_morphism(source, target, morphism)

    def get_metadata(self, text_obj: Any) -> Dict[str, Any]:
        return self.metadata_map.get(text_obj, {})

    def print_metadata(self):
        for text, meta in self.metadata_map.items():
            preview = str(text)[:30].replace('\n', ' ')
            print(f"Text: {preview}..., Metadata: {meta}")
