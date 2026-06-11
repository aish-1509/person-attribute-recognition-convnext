"""Inference utilities for the multi-task person attribute model."""

from par_model.model import MultiTaskPAR

__all__ = ["MultiTaskPAR", "PersonAttributePredictor"]
__version__ = "1.0.0"


def __getattr__(name: str):
    if name == "PersonAttributePredictor":
        from par_model.predict import PersonAttributePredictor

        return PersonAttributePredictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
