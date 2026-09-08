# NFL Draft Prediction

Predicting whether a college football player will be drafted into the NFL, using combine performance metrics and player attributes.

## Problem

Given a player's physical measurements, position, and athletic testing results, predict `Drafted` (1 = drafted, 0 = not drafted). Training data: 2,781 players / 16 features. Test data: 696 players. Baseline class split: ~65% drafted, ~35% not drafted.

## Approach

- **EDA:** examined missing data, class balance, feature correlations, and draft rate by position/player type — found that special-teams and kicking-specialist players have notably lower draft rates.
- **Preprocessing:** dropped `Id` and `School` (236 levels, too sparse to encode usefully); added missing-value flags; clipped outliers to the 1st–99th percentile; imputed remaining gaps with `IterativeImputer` (MICE) instead of simple median fill.
- **Feature engineering:** derived athleticism metrics (BMI, Speed Score, Burst Score, Explosion Index, Agility Composite), pairwise interaction features (Speed × Explosion, Size × Speed, etc.), position-relative percentile ranks for key combine tests, and target-encoded draft rates by position/year (computed fold-safe to avoid leakage).
- **Modeling:** 5-fold stratified cross-validation with a stacked ensemble — Random Forest, XGBoost, LightGBM, and CatBoost as base models, blended by a meta-model trained on out-of-fold predictions. Hyperparameters tuned with Optuna.

## Results

- Mean blended out-of-fold AUC: **0.835**
- Meta-model (stacked) out-of-fold AUC: **0.849**
- CatBoost carried the most weight in the final blend, followed by Random Forest and LightGBM
- Final predicted draft rate on the test set: **63.1%**, consistent with the training distribution


## Files

- `NFL_draft.ipynb` — full notebook: EDA, preprocessing, feature engineering, model stacking, and submission generation
- `submission.csv` — final predicted probabilities for the test set
