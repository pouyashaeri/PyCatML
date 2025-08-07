import torch

# Natural Transformation Class
class NaturalTransformation:
    def __init__(self, source_functor, target_functor, components):
        self.source = source_functor
        self.target = target_functor
        self.components = components  # dict: object → morphism

    def apply(self, obj):
        return self.components[obj]

    def verify_naturality(self, obj, morph):
        """
        Checks: η_Y ∘ F(f) == G(f) ∘ η_X
        """
        η_X = self.components[obj]
        η_Y = self.components[morph(obj)]

        Ff = self.source.apply_morph(morph)
        Gf = self.target.apply_morph(morph)

        lhs = η_Y(Ff(obj))
        rhs = Gf(η_X(obj))

        return torch.allclose(lhs, rhs) if isinstance(lhs, torch.Tensor) else lhs == rhs
