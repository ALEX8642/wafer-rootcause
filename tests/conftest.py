"""Session-scoped sim + DB fixtures built from the default config."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wafer_rootcause.config import SimConfig          # noqa: E402
from wafer_rootcause.db import build_db, connect      # noqa: E402
from wafer_rootcause.simulate import simulate         # noqa: E402


@pytest.fixture(scope="session")
def cfg() -> SimConfig:
    return SimConfig.from_yaml(REPO_ROOT / "configs" / "sim_baseline.yaml")


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
