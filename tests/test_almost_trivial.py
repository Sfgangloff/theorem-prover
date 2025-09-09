import os
import re
import sys
import shutil
import subprocess
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[1]
LEAN_ENV_DIR = REPO / "lean_env"
LEANSRC_DIR = LEAN_ENV_DIR / "LeanEnv"

PROVER = REPO / "two_stages_auto_prove.py"
TEMPLATE_CKPT = Path(os.getenv("TEMPLATE_CKPT", REPO / "runs" / "template" / "template.pt"))
LEMMAS_CKPT   = Path(os.getenv("LEMMAS_CKPT", ""))  # optional

def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None

@pytest.mark.skipif(not have("lake"), reason="requires Lean/lake on PATH")
@pytest.mark.skipif(not PROVER.exists(), reason="two_stages_auto_prove.py not found")
@pytest.mark.skipif(not TEMPLATE_CKPT.exists(), reason="set TEMPLATE_CKPT to trained template checkpoint")
def test_two_stage_simple_proof(tmp_path):
    # Arrange: write a fresh mini template in LeanEnv/
    tpl = LEANSRC_DIR / "ExampleTest.lean"
    out = LEANSRC_DIR / "ExampleTest.proof.lean"
    tpl.write_text(
        (
            "open Classical\n"
            "theorem test_and_comm (p q : Prop) : p ∧ q ↔ q ∧ p := by\n"
            "  -- @TACTICS@\n"
        ),
        encoding="utf-8",
    )

    # Build command targets the *output* proof file
    build_cmd = f"lake env lean {out.as_posix()}"

    # Run prover
    cmd = [
        sys.executable, str(PROVER),
        "--ckpt_template", str(TEMPLATE_CKPT),
        # omit --ckpt_lemmas: this example needs only structural steps
        "--template", str(tpl),
        "--out", str(out),
        "--project_root", str(LEAN_ENV_DIR),
        "--build", build_cmd,
        "--decl", "test_and_comm",
        "--topk_templates", "5",
        "--topk_lemmas", "3",
        "--max_steps", "20",
        "--wait_s", "0.4",
        "--verbose",
    ]
    if LEMMAS_CKPT.exists():
        cmd.extend(["--ckpt_lemmas", str(LEMMAS_CKPT)])

    run = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True)
    print("STDOUT:\n", run.stdout)
    print("STDERR:\n", run.stderr)
    assert run.returncode == 0, "prover process failed"

    # Assert success: True in summary
    assert re.search(r"^success:\s*True\s*$", run.stdout, flags=re.M) is not None, "prover did not succeed"

    # The produced file should have no wrappers after cleanup
    txt = out.read_text(encoding="utf-8")
    assert "logStepAuto" not in txt, "wrappers not stripped"
    assert "theorem test_and_comm" in txt, "theorem name missing in output"

    # Final sanity: the output file compiles cleanly with lake/lean
    lake = subprocess.run(["bash", "-lc", build_cmd], cwd=LEAN_ENV_DIR, text=True, capture_output=True)
    print("LAKE STDOUT:\n", lake.stdout)
    print("LAKE STDERR:\n", lake.stderr)
    assert lake.returncode == 0, "lean build failed for the produced proof"

    # Cleanup: keep files if you want artifacts; otherwise uncomment next line
    # tpl.unlink(missing_ok=True); out.unlink(missing_ok=True)