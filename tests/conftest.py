"""Session-scoped sim + DB fixtures built from the default config."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wafer_rootcause.attach import (assign_maps, load_test_inventory,  # noqa: E402
                                    record_assignment, wafer_combos)
from wafer_rootcause.config import AttachConfig, SimConfig  # noqa: E402
from wafer_rootcause.db import build_db, connect, read_parquet  # noqa: E402
from wafer_rootcause.predict import load_classifier_outputs  # noqa: E402
from wafer_rootcause.simulate import simulate         # noqa: E402


@pytest.fixture(scope="session")
def cfg() -> SimConfig:
    return SimConfig.from_yaml(REPO_ROOT / "configs" / "sim_baseline.yaml")


@pytest.fixture(scope="session")
def acfg() -> AttachConfig:
    return AttachConfig.from_yaml(REPO_ROOT / "configs" / "attach_baseline.yaml")


@pytest.fixture(scope="session")
def inventory(acfg):
    """Test-split map inventory; skips when the wafer-mixed checkout is absent."""
    for needed in (acfg.npz_path, acfg.splits_path):
        if not needed.exists():
            pytest.skip(f"wafer-mixed data not available ({needed})")
    return load_test_inventory(acfg)


@pytest.fixture(scope="session")
def assignment(db, inventory, acfg):
    return assign_maps(wafer_combos(db), inventory, acfg.assign_seed)


@pytest.fixture(scope="session")
def cache(acfg):
    """The repo's prediction cache; skips when it hasn't been built yet."""
    cache_path = REPO_ROOT / acfg.cache_path
    if not cache_path.exists():
        pytest.skip(f"prediction cache not built ({cache_path}) — "
                    "run scripts/attach_and_predict.py")
    return read_parquet(cache_path)


@pytest.fixture(scope="session")
def attached_db(tables, assignment, cache, tmp_path_factory):
    """Fresh DB with Phase 1 tables loaded from the repo's prediction cache."""
    db_path = tmp_path_factory.mktemp("db") / "attached.duckdb"
    build_db(tables, db_path)
    con = connect(db_path, read_only=False)
    record_assignment(con, assignment)
    load_classifier_outputs(con, assignment, cache)
    yield con
    con.close()


@pytest.fixture(scope="session")
def tables(cfg):
    return simulate(cfg)


@pytest.fixture(scope="session")
def db(tables, tmp_path_factory):
    db_path = tmp_path_factory.mktemp("db") / "test.duckdb"
    build_db(tables, db_path)
    con = connect(db_path)
    yield con
    con.close()
