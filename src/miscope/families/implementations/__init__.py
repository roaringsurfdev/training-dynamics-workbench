"""Family-specific implementations.

This module contains concrete ModelFamily implementations that provide
the create_model() and generate_analysis_dataset() methods.
"""

from miscope.families.implementations.modulo_addition_1layer import ModuloAddition1LayerFamily

__all__ = ["ModuloAddition1LayerFamily"]
