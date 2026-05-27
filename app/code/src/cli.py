import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

from check_required_files import RequiredFilesError, check_required_files
from load_submission_config import build_default_inference_args, load_submission_config


ORIGINAL_CWD = Path.cwd()


def default_app_root() -> Path:
    if Path("/app/code").is_dir():
        return Path("/app")
    return Path(__file__).resolve().parents[2]


def env_value(name: str, default: object) -> str:
    value = os.environ.get(name)
    return str(value) if value not in (None, "") else str(default)


def env_optional(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def resolve_user_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (ORIGINAL_CWD / path).resolve()


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"[cli] run: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def paths(app_root: Path) -> dict[str, Path]:
    return {
        "app_root": app_root,
        "package_root": app_root.parent if app_root != Path("/app") else Path("/"),
        "code_root": app_root / "code",
        "src_root": app_root / "code" / "src",
        "data_dir": app_root / "data",
        "temp_dir": app_root / "temp",
        "model_dir": app_root / "model",
        "output_dir": app_root / "output",
    }


def python_bin() -> str:
    return os.environ.get("PYTHON_BIN") or sys.executable or "python"


def run_init(app_root: Path) -> None:
    if platform.system().lower().startswith("win"):
        run(["powershell", "-ExecutionPolicy", "Bypass", "-File", str(app_root / "init.ps1")])
    else:
        run(["bash", str(app_root / "init.sh")])


def latest_experiment_dir(experiment_root: Path) -> Path:
    candidates = [path for path in experiment_root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No experiment directory found under {experiment_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def train(args: argparse.Namespace, extra_args: list[str]) -> None:
    p = paths(args.app_root)
    py = python_bin()
    experiment_root = resolve_user_path(env_value("EXPERIMENT_ROOT", p["model_dir"] / "experiments"))

    check_required_files(p["app_root"], "train")
    run_init(p["app_root"])
    run(
        [
            py,
            str(p["src_root"] / "featurework.py"),
            "--mode",
            "train",
            "--data_dir",
            str(p["data_dir"]),
            "--temp_dir",
            str(p["temp_dir"]),
        ],
        cwd=p["code_root"],
    )
    run(
        [
            py,
            str(p["src_root"] / "train_lstm.py"),
            "--feature_path",
            str(p["temp_dir"] / "train_features.csv"),
            "--model_dir",
            str(p["model_dir"]),
            "--experiment_root",
            str(experiment_root),
            "--experiment_id",
            env_optional("EXPERIMENT_ID"),
            "--experiment_remark",
            env_value("EXPERIMENT_REMARK", "train"),
            "--feature_set",
            env_value("FEATURE_SET", "base_alpha_v3_rs_crowding_mini4"),
            "--target_mode",
            env_value("TARGET_MODE", "cross_section_rank"),
            "--sequence_length",
            env_value("SEQUENCE_LENGTH", 10),
            "--hidden_size",
            env_value("HIDDEN_SIZE", 64),
            "--num_layers",
            env_value("NUM_LAYERS", 1),
            "--dropout",
            env_value("DROPOUT", 0.0),
            "--learning_rate",
            env_value("LEARNING_RATE", 0.001),
            "--batch_size",
            env_value("BATCH_SIZE", 256),
            "--epochs",
            env_value("EPOCHS", 8),
            "--patience",
            env_value("PATIENCE", 2),
            *extra_args,
        ],
        cwd=p["code_root"],
    )

    latest_dir = latest_experiment_dir(experiment_root)
    run_backtest(
        argparse.Namespace(
            app_root=p["app_root"],
            prediction_path=latest_dir / "walk_forward_predictions.csv",
            feature_path=p["temp_dir"] / "train_features.csv",
            model_dir=latest_dir,
            output_dir=latest_dir,
        ),
        [],
    )
    print("[cli] training pipeline completed.")


def predict(args: argparse.Namespace, extra_args: list[str]) -> None:
    p = paths(args.app_root)
    py = python_bin()
    check_required_files(p["app_root"], "predict")
    config_path = env_optional("SUBMISSION_CONFIG_PATH") or env_optional("CANDIDATE_CONFIG_PATH")
    submission_config = load_submission_config(config_path) if config_path else load_submission_config()
    defaults = build_default_inference_args(submission_config)
    output_path = resolve_user_path(env_value("OUTPUT_PATH", p["output_dir"] / "result.csv"))
    history_feature_path = resolve_user_path(env_value("HISTORY_FEATURE_PATH", p["temp_dir"] / "train_features.csv"))
    previous_result_path = env_optional("PREVIOUS_RESULT_PATH")
    auto_previous = env_value("AUTO_USE_PREVIOUS_RESULT", defaults["auto_use_previous_result"])
    score_output_path = resolve_user_path(env_value("SCORE_OUTPUT_PATH", p["output_dir"] / "predict_scores.csv"))
    debug_candidates_path = resolve_user_path(env_value("DEBUG_CANDIDATES_PATH", p["output_dir"] / "debug_candidates.csv"))

    run_init(p["app_root"])
    if auto_previous == "1" and not previous_result_path and output_path.is_file():
        previous_result_path = str(output_path)
        print(f"[cli] auto-detected previous result: {previous_result_path}")
    if previous_result_path and not Path(previous_result_path).is_file():
        raise RequiredFilesError(
            f"[ERROR] Missing {previous_result_path}\n"
            "        Fix: set PREVIOUS_RESULT_PATH to an existing result.csv, or unset it to run without old-result turnover control."
        )

    run(
        [
            py,
            str(p["src_root"] / "featurework.py"),
            "--mode",
            "predict",
            "--data_dir",
            str(p["data_dir"]),
            "--temp_dir",
            str(p["temp_dir"]),
        ],
        cwd=p["code_root"],
    )
    if not history_feature_path.is_file():
        print(f"[cli] history features missing, generating {history_feature_path}")
        run(
            [
                py,
                str(p["src_root"] / "featurework.py"),
                "--mode",
                "train",
                "--data_dir",
                str(p["data_dir"]),
                "--temp_dir",
                str(p["temp_dir"]),
            ],
            cwd=p["code_root"],
        )

    cmd = [
        py,
        str(p["src_root"] / "test_lstm.py"),
        "--feature_path",
        str(p["temp_dir"] / "predict_features.csv"),
        "--model_dir",
        str(p["model_dir"]),
        "--output_path",
        str(output_path),
        "--history_feature_path",
        str(history_feature_path),
        "--score_output_path",
        str(score_output_path),
        "--debug_candidates_path",
        str(debug_candidates_path),
        "--top_k",
        env_value("TOP_K", defaults["top_k"]),
        "--primary_candidate_size",
        env_value("PRIMARY_CANDIDATE_SIZE", defaults["primary_candidate_size"]),
        "--max_volatility_20d_pct",
        env_value("MAX_VOLATILITY_20D_PCT", defaults["max_volatility_20d_pct"]),
        "--max_volatility_5d_pct",
        env_value("MAX_VOLATILITY_5D_PCT", defaults["max_volatility_5d_pct"]),
        "--turnover_rate_lower_pct",
        env_value("TURNOVER_RATE_LOWER_PCT", defaults["turnover_rate_lower_pct"]),
        "--turnover_rate_upper_pct",
        env_value("TURNOVER_RATE_UPPER_PCT", defaults["turnover_rate_upper_pct"]),
        "--turnover_ratio_upper_pct",
        env_value("TURNOVER_RATIO_UPPER_PCT", defaults["turnover_ratio_upper_pct"]),
        "--risk_penalty_weight",
        env_value("RISK_PENALTY_WEIGHT", defaults["risk_penalty_weight"]),
        "--sort_strategy",
        env_value("SORT_STRATEGY", defaults["sort_strategy"]),
        "--weighting_scheme",
        env_value("WEIGHTING_SCHEME", defaults["weighting_scheme"]),
        "--weight_blend_alpha",
        env_value("WEIGHT_BLEND_ALPHA", defaults["weight_blend_alpha"]),
        "--max_single_weight",
        env_value("MAX_SINGLE_WEIGHT", defaults["max_single_weight"]),
        "--max_turnover",
        env_value("MAX_TURNOVER", defaults["max_turnover"]),
    ]
    rerank_signal_column = env_value("RERANK_SIGNAL_COLUMN", defaults.get("rerank_signal_column", ""))
    rerank_signal_weight = env_value("RERANK_SIGNAL_WEIGHT", defaults.get("rerank_signal_weight", 0.0))
    if rerank_signal_column:
        cmd.extend(["--rerank_signal_column", rerank_signal_column])
        cmd.extend(["--rerank_signal_weight", rerank_signal_weight])
    regime_rerank_enabled = env_value(
        "REGIME_RERANK_ENABLED",
        int(bool(defaults.get("regime_rerank_enabled", False))),
    )
    if regime_rerank_enabled == "1":
        cmd.extend(["--regime_rerank_enabled"])
        cmd.extend(["--regime_rerank_flag", env_value("REGIME_RERANK_FLAG", defaults.get("regime_rerank_flag", ""))])
        cmd.extend(["--regime_rerank_signal", env_value("REGIME_RERANK_SIGNAL", defaults.get("regime_rerank_signal", ""))])
        cmd.extend(["--regime_rerank_weight", env_value("REGIME_RERANK_WEIGHT", defaults.get("regime_rerank_weight", 0.0))])
    if previous_result_path:
        cmd.extend(["--previous_result_path", previous_result_path])
    cmd.extend(extra_args)
    run(cmd, cwd=p["code_root"])

    validate(argparse.Namespace(app_root=p["app_root"], result_path=output_path), [])
    run(
        [
            py,
            str(p["src_root"] / "analyze_position_concentration.py"),
            "--result_path",
            str(output_path),
            "--price_path",
            str(p["data_dir"] / "test.csv"),
            "--output_dir",
            str(p["output_dir"]),
        ],
        cwd=p["code_root"],
    )
    print("[cli] inference pipeline completed.")


def validate(args: argparse.Namespace, extra_args: list[str]) -> None:
    p = paths(args.app_root)
    result_path = resolve_user_path(args.result_path or p["output_dir"] / "result.csv")
    run(
        [
            python_bin(),
            str(p["src_root"] / "result_validator.py"),
            "--result_path",
            str(result_path),
            *extra_args,
        ],
        cwd=p["code_root"],
    )


def run_backtest(args: argparse.Namespace, extra_args: list[str]) -> None:
    p = paths(args.app_root)
    run(
        [
            python_bin(),
            str(p["src_root"] / "backtest.py"),
            "--prediction_path",
            str(resolve_user_path(args.prediction_path or p["model_dir"] / "walk_forward_predictions.csv")),
            "--feature_path",
            str(resolve_user_path(args.feature_path or p["temp_dir"] / "train_features.csv")),
            "--model_dir",
            str(resolve_user_path(args.model_dir or p["model_dir"])),
            "--output_dir",
            str(resolve_user_path(args.output_dir or p["model_dir"])),
            *extra_args,
        ],
        cwd=p["code_root"],
    )


def freeze(args: argparse.Namespace, extra_args: list[str]) -> None:
    p = paths(args.app_root)
    py = python_bin()
    check_required_files(p["app_root"], "freeze")
    print("[cli] sync submission config...")
    run([py, str(p["src_root"] / "sync_submission_config.py")], cwd=p["code_root"])
    print("[cli] compare config consistency...")
    run([py, str(p["src_root"] / "compare_config_consistency.py")], cwd=p["code_root"])
    print("[cli] run inference...")
    predict(argparse.Namespace(app_root=p["app_root"]), extra_args)
    print("[cli] validate result.csv...")
    validate(argparse.Namespace(app_root=p["app_root"], result_path=p["output_dir"] / "result.csv"), [])
    print("[cli] run pre-submit check...")
    run(
        [
            py,
            str(p["src_root"] / "pre_submit_check.py"),
            "--root_dir",
            str(p["package_root"]),
            "--result_path",
            "app/output/result.csv",
        ],
        cwd=p["code_root"],
    )
    print("[cli] refresh case comparison...")
    run([py, str(p["src_root"] / "build_case_program_comparison.py")], cwd=p["code_root"])
    run([py, str(p["src_root"] / "compare_with_case_score.py")], cwd=p["code_root"])
    print("[cli] submission freeze pipeline completed.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified project CLI.")
    parser.add_argument("--app-root", type=Path, default=default_app_root())
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("train", help="Run feature generation, LSTM training, and experiment backtest.")
    subparsers.add_parser("predict", help="Run frozen inference, validation, and concentration diagnostics.")
    subparsers.add_parser("freeze", help="Sync submission config, infer, validate, and run pre-submit checks.")

    validate_parser = subparsers.add_parser("validate", help="Validate result.csv format.")
    validate_parser.add_argument("--result_path")

    backtest_parser = subparsers.add_parser("backtest", help="Run local rolling backtest.")
    backtest_parser.add_argument("--prediction_path", type=Path)
    backtest_parser.add_argument("--feature_path", type=Path)
    backtest_parser.add_argument("--model_dir", type=Path)
    backtest_parser.add_argument("--output_dir", type=Path)
    return parser


def main() -> None:
    parser = build_parser()
    args, extra_args = parser.parse_known_args()
    args.app_root = args.app_root.resolve()
    handlers = {
        "train": train,
        "predict": predict,
        "freeze": freeze,
        "validate": validate,
        "backtest": run_backtest,
    }
    try:
        handlers[args.command](args, extra_args)
    except RequiredFilesError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
