# Run multimodal example
import torch
from torchvision import transforms
from pycatml.categories.image import ImageCategory
from pycatml.categories.text import TextCategory
from pycatml.categories.multimodal import ProductCategory
from pycatml.functors.base import Functor

def main():
    # --- Set up image modality
    img = torch.randn(1, 28, 28)  # dummy grayscale image
    img_cat = ImageCategory()
    img_cat.add_image_object(img, {"shape": img.shape})

    resized = transforms.Resize((32, 32))(img)
    img_cat.add_image_object(resized, {"shape": resized.shape})
    img_cat.add_image_morphism(img, resized, transforms.Resize((32, 32)))

    # --- Set up text modality
    text = "The quick brown fox."
    txt_cat = TextCategory()
    txt_cat.add_text_object(text, {"lang": "en"})

    lower_text = text.lower()
    txt_cat.add_text_object(lower_text, {"transformation": "lowercase"})
    txt_cat.add_text_morphism(text, lower_text, lambda x: x.lower())

    # --- Define product category
    multi_cat = ProductCategory((img_cat, txt_cat))
    multi_cat.add_multimodal_object((img, text))
    multi_cat.add_multimodal_object((resized, lower_text))
    multi_cat.add_multimodal_morphism(
        (img, text),
        (resized, lower_text),
        (transforms.Resize((32, 32)), lambda x: x.lower())
    )

    # --- Define multimodal functor
    multi_functor = Functor(
        obj_map=lambda pair: (transforms.Resize((32, 32))(pair[0]), pair[1].lower()),
        morph_map=lambda fm: (lambda x: fm[0](x[0]), lambda y: fm[1](y[1]))
    )

    transformed_pair = multi_functor.apply_obj((img, text))
    print(f"Image shape: {transformed_pair[0].shape}")
    print(f"Text: {transformed_pair[1]}")

if __name__ == "__main__":
    main()
