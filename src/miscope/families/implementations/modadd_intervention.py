"""Deprecated: ModAddInterventionFamily (REQ_071).

Interventions are now sub-variants nested under their parent Variant:

    parent.interventions                        -> list[InterventionVariant]
    parent.create_intervention_variant(config)  -> InterventionVariant

The modadd_intervention family and registry entry have been removed.
This module is kept only to preserve the compute_intervention_id import
path for any existing notebooks.  Use the canonical location instead::

    from miscope.families.intervention_variant import compute_intervention_id
"""

from miscope.families.intervention_variant import compute_intervention_id  # noqa: F401

__all__ = ["compute_intervention_id"]
