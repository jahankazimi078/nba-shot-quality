"""CLI entrypoint for the shot-quality pipeline."""

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

    p_stats = sub.add_parser("ingest-stats", help="pull league player totals (true shooting) from nba_api")
    p_stats.add_argument("--season", required=True)
    p_stats.add_argument("--force", action="store_true", help="re-pull even if cached")

    p_score = sub.add_parser("score", help="out-of-fold xPoints over the full season")
    p_score.add_argument("--season", required=True)

    p_poe = sub.add_parser("poe", help="aggregate per-player-season POE with bootstrap CIs")
    p_poe.add_argument("--season", required=True)
    p_poe.add_argument("--min-attempts", type=int, default=200)

    p_stab = sub.add_parser("stability", help="year-over-year POE correlation")
    p_stab.add_argument("--season-a", required=True)
    p_stab.add_argument("--season-b", required=True)
    p_stab.add_argument("--min-attempts", type=int, default=200)

    p_pvr = sub.add_parser("poe-vs-rts", help="POE vs relative true-shooting scatter")
    p_pvr.add_argument("--season", required=True)
    p_pvr.add_argument("--min-attempts", type=int, default=200)

    p_rot = sub.add_parser("ingest-rotations", help="build per-game player rotations from nba_api")
    p_rot.add_argument("--season", required=True)
    p_rot.add_argument("--force", action="store_true", help="re-pull even if cached")
    p_rot.add_argument("--source", choices=["pbp", "gamerotation"], default="pbp",
                       help="pbp = fast PlayByPlayV3 reconstruction (default); gamerotation = slow exact endpoint")

    p_lineups = sub.add_parser("lineups", help="reconstruct on-floor 5v5 per shot")
    p_lineups.add_argument("--season", required=True)

    p_rapm = sub.add_parser("rapm", help="fit defender-impact ridge RAPM (pooled if multiple seasons)")
    p_rapm.add_argument("--seasons", required=True, nargs="+", help="one or more, e.g. 2023-24 2024-25")
    p_rapm.add_argument("--n-boot", type=int, default=300)
    p_rapm.add_argument("--weighting", choices=["uniform", "matchup"], default="uniform",
                        help="uniform = equal 5-way defensive credit (default); matchup = weight by who guarded the shooter")
    p_rapm.add_argument("--lam", type=float, default=1.0, help="matchup blend: 1.0 pure matchup, 0.0 = uniform")

    p_re = sub.add_parser("rapm-eval", help="defender RAPM stability + face validity")
    p_re.add_argument("--season-a", required=True)
    p_re.add_argument("--season-b", required=True)
    p_re.add_argument("--season", required=True, help="season for the face-validity comparison")
    p_re.add_argument("--min-def-shots", type=int, default=1500)

    p_wc = sub.add_parser("rapm-weighting-compare", help="uniform vs matchup-weighted reliability/stability")
    p_wc.add_argument("--season-a", required=True)
    p_wc.add_argument("--season-b", required=True)
    p_wc.add_argument("--min-def-shots", type=int, default=1500)
    p_wc.add_argument("--lam", type=float, default=1.0)

    p_def = sub.add_parser("ingest-def", help="pull tracking defended-FG stats from nba_api")
    p_def.add_argument("--season", required=True)
    p_def.add_argument("--force", action="store_true", help="re-pull even if cached")

    p_mu = sub.add_parser("ingest-matchups", help="pull season player-vs-player matchup tracking from nba_api")
    p_mu.add_argument("--season", required=True)
    p_mu.add_argument("--force", action="store_true", help="re-pull even if cached")

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
    elif args.cmd == "ingest-stats":
        from nba_shot_quality.ingest.player_stats import ingest_player_stats

        ingest_player_stats(args.season, force=args.force)
    elif args.cmd == "score":
        from nba_shot_quality.models.xpoints import score_oof

        score_oof(args.season)
    elif args.cmd == "poe":
        from nba_shot_quality.models.poe import aggregate_player_season

        aggregate_player_season(args.season, min_attempts=args.min_attempts)
    elif args.cmd == "stability":
        from nba_shot_quality.eval.poe_stability import yoy_stability

        yoy_stability(args.season_a, args.season_b, min_attempts=args.min_attempts)
    elif args.cmd == "poe-vs-rts":
        from nba_shot_quality.eval.poe_stability import poe_vs_rts

        poe_vs_rts(args.season, min_attempts=args.min_attempts)
    elif args.cmd == "ingest-rotations":
        if args.source == "pbp":
            from nba_shot_quality.ingest.pbp_rotations import ingest_pbp_rotations

            ingest_pbp_rotations(args.season, force=args.force)
        else:
            from nba_shot_quality.ingest.rotations import ingest_rotations

            ingest_rotations(args.season, force=args.force)
    elif args.cmd == "lineups":
        from nba_shot_quality.features.shot_lineups import build_shot_lineups

        build_shot_lineups(args.season)
    elif args.cmd == "rapm":
        from nba_shot_quality.models.rapm import fit_rapm

        fit_rapm(args.seasons, n_boot=args.n_boot, weighting=args.weighting, lam=args.lam)
    elif args.cmd == "rapm-eval":
        from nba_shot_quality.eval.rapm_eval import (
            ptdefend_yoy,
            rapm_face_validity,
            rapm_off_vs_poe,
            splithalf_reliability,
            yoy_rapm_stability,
        )

        yoy_rapm_stability(args.season_a, args.season_b, min_def_shots=args.min_def_shots)
        rapm_face_validity(args.season, min_def_shots=args.min_def_shots)
        rapm_off_vs_poe(args.season, min_shots=args.min_def_shots)
        ptdefend_yoy(args.season_a, args.season_b)
        splithalf_reliability(args.season, min_def_shots=args.min_def_shots)
    elif args.cmd == "rapm-weighting-compare":
        from nba_shot_quality.eval.rapm_eval import weighting_compare

        weighting_compare(args.season_a, args.season_b, min_def_shots=args.min_def_shots, lam=args.lam)
    elif args.cmd == "ingest-def":
        from nba_shot_quality.ingest.player_stats import ingest_pt_defend

        ingest_pt_defend(args.season, force=args.force)
    elif args.cmd == "ingest-matchups":
        from nba_shot_quality.ingest.matchups import ingest_matchups

        ingest_matchups(args.season, force=args.force)


if __name__ == "__main__":
    main()
