"""Every function must reproduce the SFU R implementation, evaluated live by Rscript."""

import os
import shutil
import subprocess

import jax.numpy as jnp
import jax.random as jr
import pytest

from cases import CASES, KNOWN_R_BUGS

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SFU = os.path.join(ROOT, "tools", "sfu")
EVAL_R = os.path.join(ROOT, "tools", "eval_r.R")
N = 64

pytestmark = pytest.mark.skipif(
    shutil.which("Rscript") is None,
    reason="parity needs R on PATH (module load r/4.4.0)",
)


@pytest.fixture(scope="session")
def r_reference(tmp_path_factory):
    """Sample each domain, hand the points to the R sources, read back what R returned."""
    work = tmp_path_factory.mktemp("r")
    os.makedirs(work / "x")

    points = {}
    for i, (name, f, d) in enumerate(CASES):
        lo, hi = jnp.asarray(f.domain)
        u = jr.uniform(jr.fold_in(jr.key(0), i), (N, d))
        points[name] = lo + (hi - lo) * u
        rows = (",".join(repr(v) for v in row) for row in points[name].tolist())
        (work / "x" / f"{name}.csv").write_text("\n".join(rows) + "\n")

    subprocess.run(["Rscript", EVAL_R, SFU, str(work)], check=True)

    return {
        name: (
            points[name],
            [float(v) for v in (work / "y" / f"{name}.csv").read_text().split()],
        )
        for name, _, _ in CASES
    }


@pytest.mark.parametrize("name, f, d", CASES, ids=[name for name, _, _ in CASES])
def test_matches_r(name, f, d, r_reference):
    if name in KNOWN_R_BUGS:
        pytest.skip("shekelr.R hardcodes m <- 10")
    x, y_r = r_reference[name]
    assert f(x).tolist() == pytest.approx(y_r, rel=1e-12, abs=1e-12)


def test_r_evaluated_every_case(r_reference):
    assert set(r_reference) == {name for name, _, _ in CASES}
