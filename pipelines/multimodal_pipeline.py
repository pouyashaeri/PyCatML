# Multimodal learning pipeline
import torch
from torchvision import transforms
from ..categories.image import ImageCategory
from ..categories.text import TextCategory
from ..categories.multimodal import ProductCategory
from ..functors.fusion import fusion_functor

class MultimodalPipeline:
    def __init__(self, vision_encoder, text_encoder, fusion_module):
        self.image_category = ImageCategory()
        self.text_category = TextCategory()
        self.multi_category = ProductCategory((self.image_category, self.text_category))

        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.fusion_module = fusion_module

        self.fusion_functor = fusion_functor(vision_encoder, text_encoder, fusion_module)

    def load_sample(self):
        img = torch.randn(1, 28, 28)
        text = "A handwritten digit image"

        self.image_category.add_image_object(img, {"source": "synthetic"})
        self.text_category.add_text_object(text, {"lang": "en"})

        self.multi_category.add_multimodal_object((img, text))
        return (img, text)

    def run(self):
        img, text = self.load_sample()
        fused = self.fusion_functor.apply_obj((img, text))
        return fused
