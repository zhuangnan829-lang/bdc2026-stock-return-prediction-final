import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT_DIR / "app" / "code" / "src"
sys.path.insert(0, str(SRC_ROOT))

import cli


def test_cli_exposes_required_subcommands() -> None:
    parser = cli.build_parser()
    subcommands = next(action for action in parser._actions if getattr(action, "choices", None))

    assert {"train", "predict", "freeze", "validate", "backtest"} <= set(subcommands.choices)


def test_shell_and_windows_entrypoints_delegate_to_cli() -> None:
    entrypoints = {
        "app/train.sh": " train ",
        "app/test.sh": " predict ",
        "app/freeze_submission.sh": " freeze ",
        "app/train.ps1": " train ",
        "app/test.ps1": " predict ",
        "app/freeze_submission.ps1": " freeze ",
        "app/run_train.bat": " train ",
        "app/run_test.bat": " predict ",
        "app/run_freeze_submission.bat": " freeze ",
    }

    for relative_path, command in entrypoints.items():
        text = (ROOT_DIR / relative_path).read_text(encoding="utf-8")
        assert "cli.py" in text
        assert command in text
