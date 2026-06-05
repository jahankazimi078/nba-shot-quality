# NBA Shot Quality Portfolio Brief

## Problem

Raw shooting efficiency mixes shot-making skill with shot difficulty. A player who makes difficult
pull-ups and a player who finishes open rim attempts can both look efficient, but they answer
different scouting questions. This project estimates expected points for each field-goal attempt,
then turns the residual into player-level shot-making, shot-diet profiles, and model evidence.

Live app: <https://jahankazimi078.github.io/nba-shot-quality/>

## Methods

- Built an xPoints model from public NBA shot detail: distance, angle, zone, action type, period,
  clock, and shot value.
- Used game-grouped validation and out-of-fold scoring to reduce leakage into player residuals.
- Aggregated POE, or points over expected, to player seasons with bootstrap confidence intervals.
- Added player shot-diet archetypes from shot-profile features only, keeping style separate from
  performance.
- Added RAPM diagnostics for on-floor shot-quality impact and a coaching-change
  difference-in-differences study with event-clustered intervals.
- Published a static GitHub Pages dashboard backed by committed CSV exports.

## Headline Findings

- POE/100 is meaningfully stable year to year: 2023-24 to 2024-25 qualified-player correlation is
  **r = 0.58**.
- POE and relative TS% agree but are not redundant: 2024-25 correlation is **r = 0.66**.
- 2024-25 POE/100 leaders include Ty Jerome, Nikola Jokic, and Payton Pritchard among players with
  at least 200 attempts.
- Coaching-change estimates are directionally favorable for actual defensive rating, but intervals
  cross zero. The study is useful as an uncertainty-aware causal design, not as a definitive claim.

## Technical Stack

Python 3.11, pandas, scikit-learn, LightGBM, ridge regression, bootstrap intervals, public NBA data,
static HTML/CSS/JavaScript, CSV data package, GitHub Actions, and GitHub Pages.

## Resume Bullets

- Built an end-to-end NBA xPoints pipeline with grouped validation, out-of-fold scoring, and a
  static scouting dashboard deployed through GitHub Pages.
- Designed POE, a shot-quality-adjusted shooter metric with bootstrap intervals and year-over-year
  stability validation across three NBA seasons.
- Added player archetype clustering, RAPM impact diagnostics, and a coaching-change DiD case study
  to demonstrate metric design, model validation, dashboarding, and causal-analysis judgment.

## Reviewer Takeaway

This is not just a leaderboard. The portfolio package shows the full analytics workflow: data
ingestion, feature design, model validation, uncertainty communication, user-facing dashboard design,
and reproducible CSV exports that can be audited outside the app.
