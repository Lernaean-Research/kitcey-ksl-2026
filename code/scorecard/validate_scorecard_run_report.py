import argparse
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker


EXPECTED_ROW_IDS = {
    "symmetry_conservation_class",
    "gap_protection_logic",
    "leakage_retention_geometry",
    "projector_derived_observables",
    "threshold_mechanism",
}

EXPECTED_GATE_IDS = {"UV1", "UV2", "UV3", "UV4"}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_input_path(raw_path: str, script_dir: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate

    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (script_dir / candidate).resolve()


def semantic_checks(report: dict) -> list[str]:
    errors: list[str] = []

    row_ids = [row["row_id"] for row in report["row_outcomes"]]
    gate_ids = [gate["gate_id"] for gate in report["uv_gate_outcomes"]]

    row_id_set = set(row_ids)
    gate_id_set = set(gate_ids)

    if len(row_ids) != len(row_id_set):
        errors.append("Duplicate row_id values found in row_outcomes.")
    if len(gate_ids) != len(gate_id_set):
        errors.append("Duplicate gate_id values found in uv_gate_outcomes.")
    if row_id_set != EXPECTED_ROW_IDS:
        errors.append(
            "row_outcomes must contain exactly these row_id values: "
            + ", ".join(sorted(EXPECTED_ROW_IDS))
        )
    if gate_id_set != EXPECTED_GATE_IDS:
        errors.append(
            "uv_gate_outcomes must contain exactly these gate_id values: "
            + ", ".join(sorted(EXPECTED_GATE_IDS))
        )

    triggered_gates = {
        gate["gate_id"] for gate in report["uv_gate_outcomes"] if gate["status"] == "Triggered"
    }
    declared_triggered_gates = set(report["final_decision"]["triggered_gates"])
    if triggered_gates != declared_triggered_gates:
        errors.append(
            "final_decision.triggered_gates must match the UV gates marked Triggered in uv_gate_outcomes."
        )

    based_on_rows = set(report["final_decision"]["based_on_rows"])
    if not based_on_rows.issubset(row_id_set):
        errors.append("final_decision.based_on_rows contains row IDs not present in row_outcomes.")

    failed_required_rows = {
        row["row_id"]
        for row in report["row_outcomes"]
        if row.get("required_row", True)
        and (row["calibration_status"] != "Pass" or row["held_out_status"] != "Pass")
    }
    failed_nonrequired_rows = {
        row["row_id"]
        for row in report["row_outcomes"]
        if not row.get("required_row", True)
        and (row["calibration_status"] != "Pass" or row["held_out_status"] != "Pass")
    }

    decision = report["final_decision"]["decision"]
    if decision == "Accept":
        if failed_required_rows or failed_nonrequired_rows or triggered_gates:
            errors.append("Accept requires all rows to pass and no UV gates to be triggered.")
    elif decision == "Conditional":
        if failed_required_rows:
            errors.append("Conditional is invalid when any required row fails.")
        if triggered_gates:
            errors.append("Conditional is invalid when any UV gate is triggered.")
        if not failed_nonrequired_rows:
            errors.append("Conditional requires at least one nonrequired row to fail.")
    elif decision == "Reject":
        if not failed_required_rows and not failed_nonrequired_rows and not triggered_gates:
            errors.append("Reject requires at least one failed row or at least one triggered UV gate.")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a KSL scorecard run report against the JSON schema and decision rules."
    )
    parser.add_argument(
        "report",
        nargs="?",
        default="scorecard_run_report.example.json",
        help="Path to the report JSON file. Defaults to scorecard_run_report.example.json.",
    )
    parser.add_argument(
        "--schema",
        default="scorecard_run_report.schema.json",
        help="Path to the JSON Schema file. Defaults to scorecard_run_report.schema.json.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    report_path = resolve_input_path(args.report, script_dir)
    schema_path = resolve_input_path(args.schema, script_dir)

    try:
        report = load_json(report_path)
        schema = load_json(schema_path)
    except FileNotFoundError as exc:
        print(f"File not found: {exc.filename}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON in {exc.doc}: {exc}", file=sys.stderr)
        return 2

    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    schema_errors = sorted(validator.iter_errors(report), key=lambda err: list(err.path))
    if schema_errors:
        print("Schema validation failed:", file=sys.stderr)
        for err in schema_errors:
            location = "/".join(str(part) for part in err.path) or "<root>"
            print(f"- {location}: {err.message}", file=sys.stderr)
        return 1

    semantic_errors = semantic_checks(report)
    if semantic_errors:
        print("Semantic validation failed:", file=sys.stderr)
        for err in semantic_errors:
            print(f"- {err}", file=sys.stderr)
        return 1

    print(f"Validated report: {report_path.name}")
    print(f"Schema: {schema_path.name}")
    print("Result: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())