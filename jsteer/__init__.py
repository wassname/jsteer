"""jsteer: fit a model's full Jacobian once, then steer any word or persona.

    from jsteer import Jacobian
    jac = Jacobian.fit(model, tok, prompts)         # expensive, once, cacheable
    v = jac.word_vector(model, tok, ["authority"])  # instant matvec
    with v(model, C=8):                             # steering-lite runtime
        model.generate(**inputs)
"""
from . import applies  # noqa: F401  -- registers methods into steering-lite's REGISTRY
from .demo import chat_input, show_steer, split_think
from .jacobian import Jacobian
from .vjp import pullback_vjp, word_vector_vjp

__all__ = ["Jacobian", "pullback_vjp", "word_vector_vjp",
           "show_steer", "chat_input", "split_think"]
