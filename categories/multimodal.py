# Multimodal (Product) Category
from typing import Tuple, Callable, Any
from .base import Category

class ProductCategory(Category):
    def __init__(self, categories: Tuple[Category, ...]):
        super().__init__()
        self.subcategories = categories

    def add_multimodal_object(self, multimodal_obj: Tuple[Any, ...]):
        for i, obj in enumerate(multimodal_obj):
            self.subcategories[i].add_object(obj)
        self.objects.add(multimodal_obj)

    def add_multimodal_morphism(self, source_obj: Tuple[Any, ...], target_obj: Tuple[Any, ...],
                                 multimodal_morphism: Tuple[Callable, ...]):
        if len(source_obj) != len(multimodal_morphism) or len(source_obj) != len(target_obj):
            raise ValueError("Mismatch in tuple lengths.")

        for i, (src, tgt, morph) in enumerate(zip(source_obj, target_obj, multimodal_morphism)):
            self.subcategories[i].add_morphism(src, tgt, morph)

        self.morphisms[(source_obj, target_obj)] = multimodal_morphism

    def apply_morphism(self, morphism: Tuple[Callable, ...], multimodal_input: Tuple[Any, ...]) -> Tuple[Any, ...]:
        return tuple(m(f) for m, f in zip(morphism, multimodal_input))
