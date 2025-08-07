# Base Category definition

from typing import Any, Callable, Dict, Tuple

class Category:
    def __init__(self):
        self.objects = set()
        self.morphisms: Dict[Tuple[Any, Any], Callable] = {}

    def add_object(self, obj: Any):
        self.objects.add(obj)

    def add_morphism(self, source: Any, target: Any, morphism: Callable):
        if source not in self.objects or target not in self.objects:
            raise ValueError("Both source and target must be valid objects in the category.")
        self.morphisms[(source, target)] = morphism

    def get_morphism(self, source: Any, target: Any) -> Callable:
        return self.morphisms.get((source, target), None)

    def identity(self, obj: Any) -> Callable:
        if obj not in self.objects:
            raise ValueError("Object must be in the category.")
        return lambda x: x

    def compose(self, f: Callable, g: Callable) -> Callable:
        return lambda x: f(g(x))
