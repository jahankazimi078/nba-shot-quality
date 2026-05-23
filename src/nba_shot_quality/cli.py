"""CLI entrypoint for the Layer 1 pipeline."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="nba_shot_quality")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="pull shot data from nba_api")
    p_ingest.add_argument("--season", required=True, help="e.g. 2024-25")
    p_ingest.add_argument("--force", action="store_true", help="re-pull even if cached")

    p_feat = sub.add_parser("features", help="build feature parquet")
    p_feat.add_argument("--season", required=True)

    p_train = sub.add_parser("train", help="train xPoints model")
    p_train.add_argument("--season", required=True)

    p_eval = sub.add_parser("eval", help="evaluate calibration on holdout")
    p_eval.add_argument("--season", required=True)

    args = parser.parse_args()
    if args.cmd == "ingest":
        from nba_shot_quality.ingest.shotlogs import ingest_season

        ingest_season(args.season, force=args.force)
    elif args.cmd == "features":
        from nba_shot_quality.features.shot_features import build_features

        build_features(args.season)
    elif args.cmd == "train":
        from nba_shot_quality.models.xpoints import train

        train(args.season)
    elif args.cmd == "eval":
        from nba_shot_quality.eval.calibration import evaluate

        evaluate(args.season)


if __name__ == "__main__":
    main()
