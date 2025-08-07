# Explanation Functor
class ExplanationFunctor:
    def __init__(self, model_category, explanation_fn):
        """
        explanation_fn: takes model representation → explanation
        """
        self.model_category = model_category
        self.explanation_fn = explanation_fn

    def apply_obj(self, model_obj):
        return self.explanation_fn(model_obj)

    def apply_morph(self, morphism):
        """
        Optional: propagate morphism through explanation mapping
        """
        return lambda x: self.explanation_fn(morphism(x))
