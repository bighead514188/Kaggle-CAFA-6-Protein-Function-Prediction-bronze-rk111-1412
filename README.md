# Kaggle-MABE-Mouse-Behavior-Estimation-bronze-rk111-1412

Kaggle Bronze Medal (111/1412) — mouse social behavior recognition from keypoint trajectories using feature engineering + tree model ensemble (LGBM/XGB/CatBoost) with multi-lab generalization.

## Approach

- **Feature Engineering**: Hand-crafted geometric features (body pose, joint angles), velocity/acceleration features (FPS-normalized), and pairwise interaction features (distance, approach/retreat, synchrony)
- **Tree Ensemble**: LGBM / XGBoost / CatBoost with StratifiedSubsetClassifier for class imbalance; multi-model fusion with varied depth, learning rate, and n_estimators
- **Post-processing**: Rolling mean temporal smoothing → continuous segment extraction → short event filtering (<3 frames) → behavior-interval submission

## Files

| File | Description |
|------|-------------|
| `mabe.ipynb` | Final solution (feature engineering + tree ensemble) |
| `mabe-f-beta.ipynb` | Evaluation metric (F-beta computation) |
| `Readme.txt` | Full project documentation (Chinese) |

## Competition

[Kaggle - MABe Mouse Behavior Detection](https://www.kaggle.com/competitions/mabe-mouse-behavior-detection)
