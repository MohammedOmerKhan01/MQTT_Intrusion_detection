"""
tune.py - Systematic Hyperparameter Tuning for Random Forest Classifier

Executes RandomizedSearchCV strictly within the training set using Stratified K-Fold.
"""

from typing import Dict, Any, Optional
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 8,
    cv_folds: int = 3,
    sample_limit: Optional[int] = 30000,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Search hyperparameter space using RandomizedSearchCV on training partition.

    Args:
        X_train: Training features
        y_train: Training target labels
        n_iter: Number of parameter combinations to sample
        cv_folds: Cross-validation folds
        sample_limit: Optional subsample to bound tuning compute time
        random_state: Random seed for reproducibility

    Returns:
        Dict containing best_estimator, best_params, and cv_results.
    """
    if sample_limit is not None and len(X_train) > sample_limit:
        X_sub = X_train.sample(n=sample_limit, random_state=random_state)
        y_sub = y_train.loc[X_sub.index]
    else:
        X_sub = X_train
        y_sub = y_train

    param_distributions = {
        "n_estimators": [50, 100, 150],
        "max_depth": [10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
        "class_weight": [None, "balanced"]
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    rf_base = RandomForestClassifier(random_state=random_state, n_jobs=-1)

    search = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="f1",
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_sub, y_sub)

    return {
        "best_estimator": search.best_estimator_,
        "best_params": search.best_params_,
        "best_f1_score": float(search.best_score_),
        "sample_size": len(X_sub)
    }
