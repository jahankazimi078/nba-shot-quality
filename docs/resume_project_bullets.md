# Resume Project Bullets

## NYC 311 City Complaints Intelligence System

- Built an end-to-end NYC 311 analytics pipeline storing raw data in SQLite and yearly Parquet files, then cleaning and QA-ing complaint records for a **2025-2026** analysis window.

- Developed a rule-based NLP taxonomy to convert complaint descriptions into issue families, issue subtypes, resolution outcomes, and model-ready predictive features.

- Trained multiclass resolution-time classifiers on **200K+ complaint records** using HistGradientBoosting and Logistic Regression, evaluating performance with feature importance, error slices, and confusion matrices.

- Produced command-line geospatial and operational reports, including borough-level burden maps, agency performance scorecards, and neighborhood fairness metrics to identify response-time disparities.

## NBA Shot Quality Analytics Platform

- Built an end-to-end NBA shot-quality pipeline analyzing **654,609 field-goal attempts** across the **2022-23, 2023-24, and 2024-25** seasons, generating per-shot expected points and player-level shooting impact metrics.

- Trained a **LightGBM xPoints model** with game-grouped validation and out-of-fold scoring; validated Points Over Expected as a repeatable skill with **0.58 year-over-year correlation** from 2023-24 to 2024-25.

- Created **983 player-season profiles** with shot-zone shares, 3PA rate, rim rate, average shot distance, TS%, relative TS%, POE/100, and bootstrap confidence intervals.

- Built a public dashboard with **6 analysis views** and **20 CSV-backed data/report exports**, including player comparison, shot maps, archetype clustering, RAPM diagnostics, and coaching-change analysis.
