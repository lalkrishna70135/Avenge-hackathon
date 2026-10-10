import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from abtest.loader import load  # noqa: E402


@pytest.fixture(scope="session")
def ds():
    return load()


@pytest.fixture(scope="session")
def db_path(ds, tmp_path_factory):
    """A results database with the win / harm / null / hold runs, separate from runs/ab.db."""
    from abtest import store
    from abtest.config import load_scenario
    from abtest.loader import ROOT
    from abtest.runner import run_experiment
    path = tmp_path_factory.mktemp("db") / "ab.db"
    con = store.connect(path)
    for name in ("win", "harm", "null", "hold"):
        run_experiment(load_scenario(ROOT / "scenarios" / f"{name}.yaml"), ds, con=con)
    con.close()
    return path
