"""
Batch validator for KSL scorecard run reports.
Validates all matching JSON files in a target directory (or an explicit list of files)
against the scorecard schema and decision rules, then prints a summary table and exits
non-zero if any report fails.

Usage
-----
  python validate_scorecard_batch.py                          # scan framework dir
  python validate_scorecard_batch.py reports/                 # scan a specific dir
  python validate_scorecard_batch.py rep1.json rep2.json ...  # explicit list
  python validate_scorecard_batch.py --schema path/to/schema.json ...
"""
import argparse
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

from validate_scorecard_run_report import resolve_input_path, semantic_checks

SCHEMA_DEFAULT = "scorecard_run_report.schema.json"
REPORT_GLOB = "scorecard_run_report*.json"
SKIP_FILES = {"scorecard_run_report.schema.json"}


def validate_one(report_path: Path, validator: Draft202012Validator) -> list[str]:
    """Return a list of error strings; empty means PASS."""
    try:
        with report_path.open("r", encoding="utf-8") as fh:
            report = json.load(fh)
    except json.JSONDecodeError as exc:
        return [f"Invalid JSON: {exc}"]

    schema_errors = sorted(validator.iter_errors(report), key=lambda e: list(e.path))
    if schema_errors:
        return [
            f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}"
            for e in schema_errors
        ]

    return semantic_checks(report)


def collect_reports(targets: list[str], script_dir: Path) -> list[Path]:
    paths: list[Path] = []
    if not targets:
        paths = [p for p in script_dir.glob(REPORT_GLOB) if p.name not in SKIP_FILES]
    else:
        for t in targets:
            p = resolve_input_path(t, script_dir)
            if p.is_dir():
                paths.extend(c for c in p.glob(REPORT_GLOB) if c.name not in SKIP_FILES)
            else:
                paths.append(p)
    return sorted(set(paths))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch-validate KSL scorecard run reports against the schema and decision rules."
    )
    parser.add_argument(
        "targets",
        nargs="*",
        help="Report JSON files or directories to scan. Defaults to the framework directory.",
    )
    parser.add_argument(
        "--schema",
        default=SCHEMA_DEFAULT,
        help=f"Path to the JSON Schema file. Defaults to {SCHEMA_DEFAULT}.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    schema_path = resolve_input_path(args.schema, script_dir)

    try:
        with schema_path.open("r", encoding="utf-8") as fh:
            schema = json.load(fh)
    except FileNotFoundError:
        print(f"Schema not found: {schema_path}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"Invalid schema JSON: {exc}", file=sys.stderr)
        return 2

    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    reports = collect_reports(args.targets, script_dir)

    if not reports:
        print("No report files found.", file=sys.stderr)
        return 2

    col_w = max(len(p.name) for p in reports) + 2
    header = f"{'Report':<{col_w}}  Result   Errors"
    print(header)
    print("-" * max(len(header), 60))

    passed = 0
    failed = 0
    for report_path in reports:
        errors = validate_one(report_path, validator)
        if errors:
            failed += 1
            print(f"{report_path.name:<{col_w}}  FAIL")
            for err in errors:
                print(f"  {'':>{col_w}}- {err}")
        else:
            passed += 1
            print(f"{report_path.name:<{col_w}}  PASS")

    print()
    print(f"Total: {len(reports)}  Passed: {passed}  Failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
