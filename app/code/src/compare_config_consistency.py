import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from load_submission_config import (
    DEFAULT_SUBMISSION_CONFIG_PATH,
    ROOT_DIR,
    build_best_config_from_submission,
    build_default_inference_args,
    build_unified_submission_config,
    load_submission_config,
)


BEST_CONFIG_PATH = ROOT_DIR / "app" / "model" / "best_config.json"
MODEL_META_PATH = ROOT_DIR / "app" / "model" / "model_meta.json"
RUN_SUBMISSION_PATH = ROOT_DIR / "app" / "run_submission.sh"
FREEZE_SUBMISSION_PATH = ROOT_DIR / "app" / "freeze_submission.sh"
TEST_SH_PATH = ROOT_DIR / "app" / "test.sh"
TEST_PS1_PATH = ROOT_DIR / "app" / "test.ps1"
CLI_PATH = ROOT_DIR / "app" / "code" / "src" / "cli.py"
REPORT_PATH = ROOT_DIR / "app" / "model" / "config_consistency_report.md"


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return bool(left) == bool(right)
    if isinstance(left, (int, float)) or isinstance(right, (int, float)):
        try:
            return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return left == right


def add_check(checks: list[dict[str, Any]], name: str, left: Any, right: Any) -> None:
    checks.append(
        {
            "name": name,
            "ok": values_equal(left, right),
            "expected": left,
            "actual": right,
        }
    )


def add_bool_check(checks: list[dict[str, Any]], name: str, ok: bool, detail: str = "") -> None:
    checks.append(
        {
            "name": name,
            "ok": bool(ok),
            "expected": "ok",
            "actual": detail or ("ok" if ok else "failed"),
        }
    )


def build_checks(
    default_config: dict[str, Any],
    best_config: dict[str, Any],
    model_meta: dict[str, Any],
) -> list[dict[str, Any]]:
    expected_best = build_best_config_from_submission(default_config, best_config)
    meta_profile = model_meta.get("default_submission_profile", {})
    checks: list[dict[str, Any]] = []

    add_check(checks, "profile_name: default vs best", default_config["profile_name"], best_config.get("profile_name"))
    add_check(checks, "profile_name: default vs model_meta", default_config["profile_name"], meta_profile.get("profile_name"))

    for key, expected_value in expected_best["training"].items():
        if key in {"hidden_size", "num_layers", "dropout", "learning_rate", "batch_size", "epochs", "patience"}:
            continue
        add_check(checks, f"training.{key}: default vs best", expected_value, best_config.get("training", {}).get(key))
    add_check(checks, "feature_set: default vs model_meta", default_config["feature_set"], model_meta.get("feature_set"))
    add_check(checks, "target_mode: default vs model_meta", default_config["target_mode"], model_meta.get("target_mode"))
    add_check(checks, "model_family: default vs model_meta", default_config["model_family"], model_meta.get("model_family"))
    if "seed" in default_config:
        add_check(checks, "seed: default vs best", default_config["seed"], best_config.get("training", {}).get("seed"))
        add_check(checks, "seed: default vs model_meta", default_config["seed"], model_meta.get("seed"))
    add_check(
        checks,
        "sequence_length: default vs model_meta",
        default_config["validation_scheme"]["sequence_length"],
        model_meta.get("sequence_length"),
    )

    for key, expected_value in default_config["selection_logic"].items():
        add_check(checks, f"selection.{key}: default vs best", expected_value, best_config.get("selection", {}).get(key))
        add_check(checks, f"selection.{key}: default vs model_meta", expected_value, meta_profile.get(key))

    for key, expected_value in default_config["risk_filter_thresholds"].items():
        add_check(checks, f"risk.{key}: default vs best", expected_value, best_config.get("risk_filter_thresholds", {}).get(key))
        if "risk_filter_thresholds" in meta_profile:
            add_check(checks, f"risk.{key}: default vs model_meta", expected_value, meta_profile["risk_filter_thresholds"].get(key))

    for key, expected_value in default_config["execution_logic"].items():
        add_check(checks, f"execution.{key}: default vs best", expected_value, best_config.get("execution", {}).get(key))
        if key in meta_profile:
            add_check(checks, f"execution.{key}: default vs model_meta", expected_value, meta_profile.get(key))
        elif "execution_logic" in meta_profile:
            add_check(checks, f"execution.{key}: default vs model_meta", expected_value, meta_profile["execution_logic"].get(key))

    return checks


def build_script_checks(default_config: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    inference_defaults = build_default_inference_args(default_config)
    unified = build_unified_submission_config(default_config)

    add_check(checks, "loader.top_k vs unified top_k", unified["top_k"], inference_defaults["top_k"])
    add_check(
        checks,
        "loader.candidate_size vs unified candidate_size",
        unified["candidate_size"],
        inference_defaults["primary_candidate_size"],
    )
    add_check(
        checks,
        "loader.weight_strategy vs unified weight_strategy",
        unified["weight_strategy"],
        inference_defaults["weighting_scheme"],
    )
    add_check(
        checks,
        "loader.risk_penalty_weight vs unified risk_penalty_weight",
        unified["risk_penalty_weight"],
        inference_defaults["risk_penalty_weight"],
    )
    add_check(checks, "loader.max_turnover vs unified max_turnover", unified["max_turnover"], inference_defaults["max_turnover"])
    add_check(
        checks,
        "loader.transaction_cost vs unified transaction_cost",
        unified["transaction_cost"],
        inference_defaults["transaction_cost"],
    )

    script_expectations = [
        (RUN_SUBMISSION_PATH, "compare_config_consistency.py"),
        (TEST_SH_PATH, "compare_config_consistency.py"),
        (TEST_PS1_PATH, "compare_config_consistency.py"),
        (FREEZE_SUBMISSION_PATH, "compare_config_consistency.py"),
        (CLI_PATH, "build_default_inference_args"),
    ]
    for path, expected_text in script_expectations:
        text = path.read_text(encoding="utf-8")
        add_bool_check(
            checks,
            f"script.{path.name} uses config consistency guard",
            expected_text in text,
            f"missing {expected_text}",
        )

    hardcode_patterns = {
        TEST_SH_PATH: ["PRIMARY_CANDIDATE_SIZE=", "RISK_PENALTY_WEIGHT=", "WEIGHTING_SCHEME=", "MAX_TURNOVER="],
        TEST_PS1_PATH: ["primaryCandidateSize", "riskPenaltyWeight", "weightingScheme", "maxTurnover"],
    }
    for path, patterns in hardcode_patterns.items():
        text = path.read_text(encoding="utf-8")
        found = [pattern for pattern in patterns if pattern in text]
        add_bool_check(
            checks,
            f"script.{path.name} has no legacy parameter defaults",
            not found,
            "legacy hardcoded parameter token(s): " + ", ".join(found) if found else "ok",
        )
    return checks


def compare_config_consistency(
    default_config_path: str | Path = DEFAULT_SUBMISSION_CONFIG_PATH,
    best_config_path: str | Path = BEST_CONFIG_PATH,
    model_meta_path: str | Path = MODEL_META_PATH,
) -> tuple[bool, list[dict[str, Any]]]:
    default_config = load_submission_config(default_config_path)
    best_config = load_json(best_config_path)
    model_meta = load_json(model_meta_path)
    checks = build_checks(default_config, best_config, model_meta)
    checks.extend(build_script_checks(default_config))
    return all(check["ok"] for check in checks), checks


def render_report(checks: list[dict[str, Any]], ok: bool) -> str:
    failed = [check for check in checks if not check["ok"]]
    lines = [
        "# Config Consistency Report",
        "",
        f"- Status: {'PASS' if ok else 'FAIL'}",
        f"- Checks: {len(checks)}",
        f"- Failed: {len(failed)}",
        f"- Authoritative source: `app/model/default_submission_config.json`",
        "",
        "## Required Unified Fields",
        "",
        "| field | source |",
        "|---|---|",
        "| model_name | default_submission_config.model_family |",
        "| feature_set | default_submission_config.feature_set |",
        "| sequence_length | validation_scheme.sequence_length |",
        "| sort_strategy | selection_logic.sort_strategy |",
        "| weight_strategy | selection_logic.weighting_scheme |",
        "| top_k | selection_logic.top_k |",
        "| candidate_size | selection_logic.primary_candidate_size |",
        "| risk_penalty_weight | risk_filter_thresholds.risk_penalty_weight |",
        "| max_turnover | execution_logic.max_turnover |",
        "| transaction_cost | execution_logic.transaction_cost |",
        "| max_single_weight | selection_logic.max_single_weight, nullable |",
        "",
        "## Failed Checks",
        "",
    ]
    if failed:
        lines.extend(["| check | expected | actual |", "|---|---|---|"])
        for check in failed:
            lines.append(f"| {check['name']} | `{check['expected']}` | `{check['actual']}` |")
    else:
        lines.append("No inconsistencies found.")

    lines.extend(["", "## All Checks", "", "| check | status | expected | actual |", "|---|---:|---|---|"])
    for check in checks:
        lines.append(
            f"| {check['name']} | {'PASS' if check['ok'] else 'FAIL'} | "
            f"`{check['expected']}` | `{check['actual']}` |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare formal submission config consistency.")
    parser.add_argument("--default_config_path", default=str(DEFAULT_SUBMISSION_CONFIG_PATH))
    parser.add_argument("--best_config_path", default=str(BEST_CONFIG_PATH))
    parser.add_argument("--model_meta_path", default=str(MODEL_META_PATH))
    parser.add_argument("--report_path", default=str(REPORT_PATH))
    parser.add_argument("--format", choices=["text", "json"], default="text")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ok, checks = compare_config_consistency(
        default_config_path=args.default_config_path,
        best_config_path=args.best_config_path,
        model_meta_path=args.model_meta_path,
    )
    failed = [check for check in checks if not check["ok"]]
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(checks, ok), encoding="utf-8")
    if args.format == "json":
        print(json.dumps({"ok": ok, "checks": checks}, ensure_ascii=False, indent=2))
    else:
        print(f"[compare_config_consistency] checks={len(checks)} failed={len(failed)}")
        print(f"[compare_config_consistency] wrote {report_path}")
        for check in failed:
            print(
                "[compare_config_consistency] FAIL "
                f"{check['name']}: expected={check['expected']!r} actual={check['actual']!r}"
            )
        if ok:
            print("[compare_config_consistency] all checks passed")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
