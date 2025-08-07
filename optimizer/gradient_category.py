# Gradient Category
class GradientCategory:
    def __init__(self):
        self.objects = set()
        self.gradients = {}  # (source, target) → gradient function

    def add_object(self, obj):
        self.objects.add(obj)

    def add_gradient(self, source, target, grad_fn):
        self.gradients[(source, target)] = grad_fn

    def get_gradient(self, source, target):
        return self.gradients.get((source, target), None)

    def compose_gradients(self, f, g, grad_f, grad_g):
        """
        Compose gradients through chain rule:
        ∇(g ∘ f)(x) = ∇g(f(x)) * ∇f(x)
        """
        return lambda x: grad_g(f(x)) * grad_f(x)

    def pullback(self, functor, grad_output):
        """
        Implements ∇_C(L) = F* (∇_D(L)) for backpropagation through a functor.
        grad_output: function from D to gradient of loss
        """
        return lambda morph: grad_output(functor.apply_morph(morph))
