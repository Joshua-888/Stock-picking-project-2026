"""
run_monthly_update.py

CLI entry point for the monthly update pipeline.

Usage
-----
    # Run with defaults (sample data, latest available date):
    python run_monthly_update.py

    # Specify a prediction date:
    python run_monthly_update.py --date 2024-01-31

    # Use a specific data provider:
    python run_monthly_update.py --provider nasdaq_data_link

    # Force model retraining (ignore cache):
    python run_monthly_update.py --force-retrain

    # Combine options:
    python run_monthly_update.py --date 2024-01-31 --provider sample --force-retrain
"""

import argparse
import sys
from pathlib import Path

# Ensure the project root is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the monthly stock analysis update pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--date", "-d",
        metavar="YYYY-MM-DD",
        default=None,
        help="Prediction month-end date (default: latest available from price data).",
    )
    parser.add_argument(
        "--provider", "-p",
        choices=["yfinance", "sample", "nasdaq_data_link"],
        default=None,
        help="Data provider override (default: from config.yaml / .env).",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        default=False,
        help="Force model retraining, ignoring any cached training run.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of stocks in the backtest portfolio (default: 10).",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        default=False,
        help="Suppress console output (logs still written to file).",
    )
    parser.add_argument(
        "--no-gems",
        action="store_true",
        default=False,
        help="Skip Hidden Gems — run only the Core 30 large-cap universe.",
    )
    parser.add_argument(
        "--gems-n",
        type=int,
        default=None,
        help="Override number of Hidden Gems to select this month (default: from config).",
    )
    args = parser.parse_args()

    from src.workflows.monthly_update import run_monthly_update

    report = run_monthly_update(
        prediction_date = args.date,
        data_provider   = args.provider,
        force_retrain   = args.force_retrain,
        top_n_portfolio = args.top_n,
        verbose         = not args.quiet,
        include_gems    = not args.no_gems,
        gems_n_override = args.gems_n,
    )

    # Exit with non-zero status if there were errors
    if report.get("status") == "failed":
        print(f"\nUpdate FAILED. Check logs for details.")
        sys.exit(1)
    elif report.get("errors"):
        print(f"\nUpdate completed with {len(report['errors'])} error(s).")
        sys.exit(2)
    else:
        print(f"\nUpdate completed successfully in {report.get('total_time_seconds', 0):.1f}s.")
        sys.exit(0)


if __name__ == "__main__":
    main()
