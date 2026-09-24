"""WP2 point-in-time data foundation for Free Research V2.

This package owns temporal data integrity: availability semantics, deterministic
as-of joins, SEC EDGAR fundamentals, correct ROA/ROIC semantics, permanent
security identifiers, historical universe membership, price/benchmark semantics,
macro vintages, layered bronze/silver/gold storage and WP1 dataset manifests.

It performs no feature engineering, no model fitting and no scoring.
"""

from .availability import (
    AvailabilityError,
    AvailabilityPolicy,
    FUNDAMENTAL,
    MACRO,
    POLICIES,
    PRICE,
    UNIVERSE,
    UnavailableError,
    availability_mask,
    filter_available,
    is_available,
    require_available,
    to_utc_timestamp,
)
from .pit_join import (
    AsofJoinSpec,
    PitJoinError,
    asof_join,
    asof_join_at_times,
    latest_version_asof,
)
from .layers import (
    DEFAULT_ROOT,
    LAYERS,
    LayerError,
    layer_root,
    read_silver_table,
    source_fingerprint,
    write_bronze_bytes,
    write_bronze_json,
    write_gold_table,
    write_silver_table,
)
from .roa_roic import (
    DEFAULT_TAX_RATE,
    ROA_DEFINITION,
    ROIC_DEFINITION,
    ReturnMetrics,
    compute_return_metrics,
    compute_roa,
    compute_roic,
)
from .security_master import (
    AmbiguousTickerError,
    SecurityMaster,
    SecurityMasterError,
    SecurityRecord,
    TickerAssignment,
    UnknownTickerError,
    security_id_for,
)
from .universe import (
    UniverseError,
    UniverseMembership,
    UniverseStatus,
    UniverseTable,
    build_universe_from_frame,
)
from .pit_prices import (
    CorporateAction,
    PRICE_SEMANTICS,
    PitPriceError,
    actions_from_yahoo_events,
    build_pit_adjusted_frame,
    pit_return,
    validate_action_ledger,
)
from .sample_universe import SampleUniverse, SampleUniverseError, load_sample_universe
from .provenance_ledger import (
    ProvenanceLedgerError,
    list_entries,
    load_manifest,
    persist_manifest,
    verify_ledger,
)
from .certification import (
    CertificationReport,
    FamilyStatus,
    build_security_master,
    run_certification,
)

__all__ = [
    "AvailabilityError",
    "AvailabilityPolicy",
    "FUNDAMENTAL",
    "MACRO",
    "POLICIES",
    "PRICE",
    "UNIVERSE",
    "UnavailableError",
    "availability_mask",
    "filter_available",
    "is_available",
    "require_available",
    "to_utc_timestamp",
    "AsofJoinSpec",
    "PitJoinError",
    "asof_join",
    "asof_join_at_times",
    "latest_version_asof",
    "DEFAULT_ROOT",
    "LAYERS",
    "LayerError",
    "layer_root",
    "read_silver_table",
    "source_fingerprint",
    "write_bronze_bytes",
    "write_bronze_json",
    "write_gold_table",
    "write_silver_table",
    "DEFAULT_TAX_RATE",
    "ROA_DEFINITION",
    "ROIC_DEFINITION",
    "ReturnMetrics",
    "compute_return_metrics",
    "compute_roa",
    "compute_roic",
    "AmbiguousTickerError",
    "SecurityMaster",
    "SecurityMasterError",
    "SecurityRecord",
    "TickerAssignment",
    "UnknownTickerError",
    "security_id_for",
    "UniverseError",
    "UniverseMembership",
    "UniverseStatus",
    "UniverseTable",
    "build_universe_from_frame",
]
