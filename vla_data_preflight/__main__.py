import argparse
import json

from .checks import check_inputs
from .demo import run_demo
from .report import write_report


def main(argv=None):
    parser = argparse.ArgumentParser(description="LIBERO/QwenOFT config and actual-array snapshot preflight")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo")
    demo.add_argument("--root", default=".")
    check = sub.add_parser("check")
    for key in ("config", "info", "modality", "samples"):
        check.add_argument(f"--{key}", required=True)
    check.add_argument("--base-dir", default=None, help="Resolve training data_root_dir relative to this directory; default is cwd")
    check.add_argument("--output", default="results/check")
    args = parser.parse_args(argv)
    if args.command == "demo":
        reports = run_demo(args.root)
        print(json.dumps({k: r["summary"] for k, r in reports.items()}, indent=2))
        return 0
    report = check_inputs(args.config, args.info, args.modality, args.samples, args.base_dir)
    write_report(report, args.output)
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
