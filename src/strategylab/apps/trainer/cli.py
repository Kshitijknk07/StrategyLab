from __future__ import annotations

import argparse
import json

from strategylab.models.catalog import get_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="StrategyLab model training CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train a registered model")
    train.add_argument("model_name")
    train.add_argument("dataset_version")
    train.add_argument("--target-column", default=None)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a registered model")
    evaluate.add_argument("model_name")
    evaluate.add_argument("dataset_version")
    evaluate.add_argument("model_version")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    model = get_model(args.model_name)
    if args.command == "train":
        result = model.train(args.dataset_version, args.target_column)
        print(json.dumps(_jsonable(result), indent=2))
        return
    report = model.evaluate(args.dataset_version, args.model_version)
    print(report.model_dump_json(indent=2))


def _jsonable(payload):
    if isinstance(payload, dict):
        return {key: _jsonable(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_jsonable(value) for value in payload]
    if hasattr(payload, "model_dump"):
        return payload.model_dump(mode="json")
    return payload

