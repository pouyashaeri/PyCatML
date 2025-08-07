# Validation and Complexity Checks
def validate_functor_laws(F, obj, f, g):
    # Identity: F(id_X) = id_{F(X)}
    identity_pass = F.apply_morph(lambda x: x)(obj) == obj

    # Composition: F(g ∘ f) = F(g) ∘ F(f)
    composed = lambda x: g(f(x))
    F_comp = F.apply_morph(composed)(obj)

    F_g_f = F.apply_morph(g)(F.apply_morph(f)(obj))
    composition_pass = F_comp == F_g_f

    return identity_pass and composition_pass