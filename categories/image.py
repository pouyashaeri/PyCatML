# Image Category implementation
import torch
from torchvision import transforms
from typing import Callable, Tuple, Dict, Any

from .base import Category

class ImageCategory(Category):
    def __init__(self):
        super().__init__()
        self.metadata_map: Dict[Any, Dict[str, Any]] = {}

    def add_image_object(self, image_tensor: torch.Tensor, metadata: Dict[str, Any]):
        self.add_object(image_tensor)
        self.metadata_map[image_tensor] = metadata

    def add_image_morphism(self, source: torch.Tensor, target: torch.Tensor, transform_fn: Callable):
        # Optional: validate that transform(source) ≈ target
        self.add_morphism(source, target, transform_fn)

    def get_metadata(self, image_tensor: torch.Tensor) -> Dict[str, Any]:
        return self.metadata_map.get(image_tensor, {})

    def print_metadata(self):
        for img, meta in self.metadata_map.items():
            print(f"Image: {img.shape if isinstance(img, torch.Tensor) else type(img)}, Metadata: {meta}")
