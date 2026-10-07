"""WP9 prospective shadow validation (forward-only, no adaptation).

This package applies the frozen WP8 champion to prospective monthly snapshots.
It never fits, selects features, recalibrates, or reads future outcomes.
"""

from .champion import (
    FROZEN_CALIBRATION,
    FROZEN_FEATURES,
    FrozenChampion,
    load_champion,
)

__all__ = [
    "FROZEN_CALIBRATION",
    "FROZEN_FEATURES",
    "FrozenChampion",
    "load_champion",
]
