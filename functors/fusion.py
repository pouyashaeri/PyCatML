# Multimodal Fusion Functor
from .base import Functor

def fusion_functor(vision_encoder, text_encoder, fusion_module):
    return Functor(
        obj_map=lambda pair: fusion_module(vision_encoder(pair[0]), text_encoder(pair[1])),
        morph_map=lambda f: lambda pair: fusion_module(vision_encoder(f[0](pair[0])), text_encoder(f[1](pair[1])))
    )
