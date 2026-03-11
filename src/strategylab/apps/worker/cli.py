from __future__ import annotations

import argparse
import json

from strategylab.contracts import IngestionRefreshRequest
from strategylab.data.service import IngestionService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="StrategyLab batch worker CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Ingest a race session into layered storage")
    ingest.add_argument("source", choices=["fixture", "fastf1", "jolpica"])
    ingest.add_argument("season", type=int)
    ingest.add_argument("event_name")
    ingest.add_argument("circuit")
    ingest.add_argument("--fixture-path", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    service = IngestionService()
    if args.command == "ingest":
        result = service.refresh(
            IngestionRefreshRequest(
                source=args.source,
                season=args.season,
                event_name=args.event_name,
                circuit=args.circuit,
                fixture_path=args.fixture_path,
            )
        )
        print(json.dumps(result, indent=2))
