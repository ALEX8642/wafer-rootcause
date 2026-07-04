"""config.py — SimConfig dataclasses + YAML loading and validation.

Mirrors the sibling repos' config style (YAML file → dataclass, paths
anchored to REPO_ROOT). The sim config is fully declarative: route, fault
list, baseline contamination rates, seed. Same config + same seed →
identical database.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from wafer_rootcause.labels import MIXABLE

# Invariant: repo root regardless of working directory or symlinks.
REPO_ROOT = Path(__file__).resolve().parents[2]


def _yaml_mapping(cls, path: str | Path) -> dict:
    """Load a YAML mapping and reject keys the dataclass doesn't declare."""
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping, got "
                         f"{type(raw).__name__}")
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{path}: unknown config keys: {sorted(unknown)}")
    return raw


def route_chambers(route):
    """Yield (step_idx, spec, tool_id, chamber_id) in canonical order.

    Single owner of the `<STEP>-T<t>-C<c>` naming convention, shared by
    config validation and the simulator's equipment tables.
    """
    for i, spec in enumerate(route):
        for t in range(1, spec.tools + 1):
            tool_id = f"{spec.name}-T{t}"
            for c in range(1, spec.chambers_per_tool + 1):
                yield i, spec, tool_id, f"{tool_id}-C{c}"


@dataclass
class StepSpec:
    """One route step and its equipment fleet."""
    name: str                 # 'ETCH'
    process_type: str         # 'dry_etch'
    tools: int                # tools serving this step
    chambers_per_tool: int
    process_minutes: float    # nominal per-wafer process time


@dataclass
class FaultSpec:
    """A planted excursion: one chamber, one time window, one signature.

    duty_on_hours/duty_off_hours (both or neither) make the fault
    intermittent: inside the [start, start+duration] envelope, exposure
    cycles on/off starting with an on-phase at start_hour. The
    ground_truth_faults table records the envelope only — the duty cycle
    lives in the config, and the scorer's window IoU is judged against
    the envelope (documented in docs/SENSITIVITY.md).
    """
    fault_id: str             # 'F1'
    chamber: str              # chamber_id, e.g. 'ETCH-T1-C2'
    label: str                # signature label the fault elevates (mixable only)
    start_hour: float         # relative to sim_start
    duration_hours: float
    p_acquire: float          # P(exposed wafer acquires the label)
    duty_on_hours: float | None = None
    duty_off_hours: float | None = None


@dataclass
class CouplingSpec:
    """Correlated dispatch: the follower step's chamber choice follows the
    driver step's.

    With probability `strength`, a wafer that took chamber index i at the
    driver step takes chamber index i mod n_follower at the follower step
    (canonical route_chambers order); otherwise the choice is uniform.
    This is the classic false-attribution trap: wafers through a faulty
    driver chamber pile into one innocent follower chamber.
    """
    driver_step: str          # route step name, e.g. 'ETCH'
    follower_step: str        # a LATER route step, e.g. 'DEPOSITION'
    strength: float           # P(follower follows the mapping) in (0, 1]


@dataclass
class BaselineRates:
    """Everywhere-contamination, structured by MixedWM38 exclusivity groups.

    center_donut / edge are per-group draw probabilities (the drawn group
    member is picked uniformly); near_full / random_pattern apply only to
    wafers that are still clean after faults + mixable baseline, because
    those labels never mix.
    """
    center_donut: float = 0.0
    edge: float = 0.0
    loc: float = 0.0
    scratch: float = 0.0
    near_full: float = 0.0
    random_pattern: float = 0.0


@dataclass
class AttachConfig:
    """Phase 1 settings: map attachment + classifier inference.

    `wafer_mixed_root` points at a sibling checkout of the wafer-mixed repo —
    its trained checkpoint, thresholds.json, MixedWM38.npz and persisted
    split are the inputs here; nothing is copied into this repo.
    """
    wafer_mixed_root: str = "../wafer-mixed"
    checkpoint: str = "outputs/best.pt"            # relative to wafer_mixed_root
    thresholds: str = "outputs/thresholds.json"    # relative to wafer_mixed_root
    assign_seed: int = 42          # map-assignment RNG, independent of the sim seed
    batch_size: int = 64           # CPU inference batch
    num_workers: int = 2
    db_path: str = "outputs/wafer_rootcause.duckdb"
    assignment_path: str = "outputs/map_assignment.parquet"
    cache_path: str = "outputs/predictions_test.parquet"

    # Path fields stay as the YAML strings; resolve on access so a config
    # with an absolute wafer_mixed_root also works (Path anchors absorb it).
    @property
    def mixed_root(self) -> Path:
        return (REPO_ROOT / self.wafer_mixed_root).resolve()

    @property
    def checkpoint_path(self) -> Path:
        return self.mixed_root / self.checkpoint

    @property
    def thresholds_path(self) -> Path:
        return self.mixed_root / self.thresholds

    # Single owner of wafer-mixed's on-disk data layout in this repo
    # (attach.py, conftest and tests all go through these).
    @property
    def npz_path(self) -> Path:
        return self.mixed_root / "data" / "raw" / "MixedWM38.npz"

    @property
    def splits_path(self) -> Path:
        return self.mixed_root / "data" / "splits.npz"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AttachConfig":
        cfg = cls(**_yaml_mapping(cls, path))
        if cfg.batch_size < 1 or cfg.num_workers < 0:
            raise ValueError("batch_size must be >= 1 and num_workers >= 0")
        return cfg


@dataclass
class SimConfig:
    seed: int = 42
    sim_start: str | datetime = "2026-06-01 00:00:00"  # YAML may parse it either way
    product: str = "PRD-A"
    n_lots: int = 40
    wafers_per_lot: int = 25
    lot_interarrival_hours: float = 4.0   # exponential mean
    wafer_stagger_minutes: float = 2.0    # within-lot release offset
    queue_mean_minutes: float = 10.0      # exponential queue delay per step
    inspect_delay_minutes: float = 30.0   # last step end → inspection
    dispatch: str = "random"              # 'random' | 'correlated' (needs coupling)
    coupling: CouplingSpec | None = None  # required iff dispatch == 'correlated'
    db_path: str = "outputs/wafer_rootcause.duckdb"
    route: list[StepSpec] = field(default_factory=list)
    baseline: BaselineRates = field(default_factory=BaselineRates)
    faults: list[FaultSpec] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SimConfig":
        raw = _yaml_mapping(cls, path)
        raw["route"] = [StepSpec(**s) for s in raw.get("route", [])]
        raw["baseline"] = BaselineRates(**raw.get("baseline", {}))
        raw["faults"] = [FaultSpec(**f) for f in raw.get("faults", [])]
        if raw.get("coupling") is not None:
            raw["coupling"] = CouplingSpec(**raw["coupling"])
        cfg = cls(**raw)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not self.route:
            raise ValueError("route must have at least one step")
        if self.dispatch not in ("random", "correlated"):
            raise ValueError(f"Unknown dispatch policy: {self.dispatch!r}")
        if (self.dispatch == "correlated") != (self.coupling is not None):
            raise ValueError(
                "dispatch: correlated and a coupling block go together — "
                f"got dispatch={self.dispatch!r}, coupling="
                f"{'set' if self.coupling else 'absent'}")
        if self.coupling is not None:
            names = [s.name for s in self.route]
            for role in ("driver_step", "follower_step"):
                if getattr(self.coupling, role) not in names:
                    raise ValueError(
                        f"coupling.{role}={getattr(self.coupling, role)!r} "
                        "is not a route step")
            if (names.index(self.coupling.driver_step)
                    >= names.index(self.coupling.follower_step)):
                raise ValueError(
                    "coupling.driver_step must come before follower_step "
                    "in the route")
            if not 0.0 < self.coupling.strength <= 1.0:
                raise ValueError(
                    f"coupling.strength={self.coupling.strength} outside (0, 1]")
        if self.n_lots < 1 or self.wafers_per_lot < 1:
            raise ValueError("n_lots and wafers_per_lot must be >= 1")
        if self.lot_interarrival_hours <= 0:
            raise ValueError("lot_interarrival_hours must be > 0")
        for field_name in ("wafer_stagger_minutes", "queue_mean_minutes",
                           "inspect_delay_minutes"):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be >= 0")
        names = [s.name for s in self.route]
        if len(names) != len(set(names)):
            raise ValueError("route step names must be unique")
        for s in self.route:
            if s.tools < 1 or s.chambers_per_tool < 1:
                raise ValueError(
                    f"{s.name}: tools and chambers_per_tool must be >= 1")
            if s.process_minutes <= 0:
                raise ValueError(f"{s.name}: process_minutes must be > 0")
        for rate_name, rate in dataclasses.asdict(self.baseline).items():
            if not 0.0 <= rate <= 1.0:
                raise ValueError(f"baseline.{rate_name}={rate} outside [0, 1]")
        chamber_ids = {cid for *_, cid in route_chambers(self.route)}
        fault_ids = [f.fault_id for f in self.faults]
        if len(fault_ids) != len(set(fault_ids)):
            raise ValueError("fault_id values must be unique")
        for f in self.faults:
            if f.chamber not in chamber_ids:
                raise ValueError(f"{f.fault_id}: unknown chamber {f.chamber!r}")
            if f.label not in MIXABLE:
                raise ValueError(
                    f"{f.fault_id}: fault label must be mixable, got {f.label!r}"
                )
            if not 0.0 < f.p_acquire <= 1.0:
                raise ValueError(f"{f.fault_id}: p_acquire outside (0, 1]")
            if f.start_hour < 0 or f.duration_hours <= 0:
                raise ValueError(f"{f.fault_id}: bad window")
            if (f.duty_on_hours is None) != (f.duty_off_hours is None):
                raise ValueError(
                    f"{f.fault_id}: duty_on_hours and duty_off_hours must be "
                    "set together")
            if f.duty_on_hours is not None:
                if f.duty_on_hours <= 0 or f.duty_off_hours <= 0:
                    raise ValueError(f"{f.fault_id}: duty phases must be > 0")
                if f.duty_on_hours + f.duty_off_hours >= f.duration_hours:
                    raise ValueError(
                        f"{f.fault_id}: duty period must fit inside the window "
                        "at least twice — otherwise the fault is not intermittent")
