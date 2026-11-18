"""Fit tuned Extra Trees on all 80 training trips and save the final model."""
import joblib
import time
from sklearn.ensemble import ExtraTreesClassifier

from train_models import ROOT, FEATURES, load, downsample, SEED

X, y, groups, obs, _ = load()
X_fit, y_fit, _ = downsample(X, y, groups, obs, SEED)
model = ExtraTreesClassifier(
    n_estimators=160,
    max_depth=16,
    min_samples_leaf=10,
    max_features="sqrt",
    class_weight="balanced",
    n_jobs=-1,
    random_state=SEED,
)
started = time.perf_counter()
model.fit(X_fit, y_fit)
training_seconds = time.perf_counter() - started
output = ROOT / "models" / "extra_trees_candidate_model.joblib"
output.parent.mkdir(exist_ok=True)
joblib.dump({
    "model": model,
    "features": FEATURES,
    "model_name": "Extra Trees",
    "n_estimators": 160,
    "max_depth": 16,
    "min_samples_leaf": 10,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "training_seconds": training_seconds,
}, output)
print(f"Saved {output}; trained on {len(set(groups))} trips and {len(y_fit):,} sampled candidate rows in {training_seconds:.2f} seconds.")
