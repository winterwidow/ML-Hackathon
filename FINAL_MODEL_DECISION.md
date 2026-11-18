# Final Model Decision

## Selected model

Use a tuned `ExtraTreesClassifier` to score each nearby directed road segment, followed by the existing HMM/Viterbi decoder. The final configuration is `n_estimators=160`, `max_depth=16`, `min_samples_leaf=10`, `max_features="sqrt"`, and `class_weight="balanced"`. The saved artifact is `models/extra_trees_candidate_model.joblib`; the API and offline decoder must load this same artifact.

## Why this model

The end goal is a plausible road sequence, so the most relevant metric is route agreement after HMM decoding. The selection rule considered both trip-grouped cross-validation on training data and the allowed 10-trip student-test development comparison. On the student-test comparison, tuned Extra Trees + HMM reached 82.20% weighted Top-1 and 85.19% mean Top-1 by trip. These were the highest route scores among the compared models. Its weighted score was 0.65 percentage points above Logistic Regression + HMM (81.55%), approximately 122 additional correctly matched labels among 18,846 heuristic route labels.

This is a development-stage selection, not a claim of performance on an untouched benchmark: the student-test comparison informed the model choice as allowed by the project instructions. The 10 evaluator trips were not used for fitting, tuning, model selection, or these results.

## Five-model grouped-CV comparison

The primary five-fold comparison kept whole trips together. The values below are the untuned family configurations in `cross_validate_models.py`; the tuned Extra Trees configuration was separately compared on three matching GroupKFold splits during Bayesian optimization.

| Model | Observation Top-1 mean +/- std | Candidate macro F1 mean | Candidate log loss mean |
| --- | ---: | ---: | ---: |
| Logistic Regression | 75.63% +/- 2.87 pp | 0.7883 | 0.2628 |
| Extra Trees | 75.61% +/- 2.61 pp | 0.7555 | 0.3627 |
| Hist Gradient Boosting | 73.94% +/- 2.68 pp | **0.8113** | **0.2274** |
| Random Forest | 73.89% +/- 2.85 pp | 0.7877 | 0.2944 |
| Decision Tree | 70.42% +/- 3.40 pp | 0.7518 | 0.6472 |

Logistic Regression and untuned Extra Trees were effectively tied in this five-fold Top-1 comparison. Hist Gradient Boosting was clearly strongest on candidate-level macro F1 and log loss. Those metrics classify each candidate row independently; they do not measure the final sequence quality by themselves.

## Bayesian tuning

A 10-trial Gaussian-process expected-improvement search optimized observation Top-1 using three-fold `GroupKFold` on the 80 training trips only. The best Extra Trees configuration was 160 trees, depth 16, minimum leaf size 10, square-root feature sampling, and balanced class weights. It scored 75.89% +/- 2.76 pp on those folds, compared with about 75.76% for untuned Extra Trees and 75.62% for Logistic Regression on the same screening folds. The improvement is small, so describe the search as a modest improvement, not a dramatic gain.

## Student-test development comparison

All six rows were trained using the 80 training trips and evaluated against the same 10 student-test trips. Candidate misses count as incorrect in the weighted Top-1 denominators. The results summarize heuristic route labels, not independently verified ground truth.

| Model | Candidate macro F1 | Candidate log loss | Classifier Top-1 | HMM weighted Top-1 | HMM mean by trip |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.7955 | 0.2706 | 76.33% | 81.55% | 84.45% |
| Decision Tree | 0.7474 | 0.6371 | 69.74% | 76.33% | 78.33% |
| Random Forest | 0.7816 | 0.3088 | 73.36% | 80.02% | 82.53% |
| Extra Trees, untuned | 0.7517 | 0.3747 | 75.83% | 81.68% | 84.80% |
| Hist Gradient Boosting | **0.8136** | **0.2340** | 73.73% | 79.62% | 82.21% |
| Extra Trees, Bayesian tuned | 0.7495 | 0.3832 | **76.03%** | **82.20%** | **85.19%** |

The final model does not lead on candidate F1 or log loss; Hist Gradient Boosting does. Tuned Extra Trees was selected because its student-test route scores after Viterbi were highest, while its training-only grouped-CV Top-1 was competitive. If the objective changes to candidate probability quality rather than route reconstruction, Hist Gradient Boosting is a credible alternative to evaluate further.

For tuned Extra Trees, the student-test HMM raised weighted all-label Top-1 from 76.03% to 82.20%. The macro mean across trips rose from 78.80% to 85.19%, and mean path continuity was 99.50%. The 190 observations without a positive candidate (1.01% of 18,846 route-labeled observations) remain misses regardless of classifier choice.

## Training artifact and input controls

`train_final_model.py` fits the selected configuration on all 80 training trips after retaining positive candidates and sampling up to four negative candidates per labeled observation. It saves the estimator, feature list, model name, and configuration in `models/extra_trees_candidate_model.joblib`. The seven model inputs are listed in `train_models.py`: time delta, GPS speed, candidate distance, road-segment length, GPS movement bearing, road bearing, and bearing difference. Route IDs, route-alignment distances, and reference-label fields are excluded from model input.

## Limitations

Training labels come from a monotone spatial route-alignment heuristic with a 50 m threshold. It can miss observations or assign imperfect arcs around crossings, long gaps, and repeated road segments. Some training trips have large unlabeled portions. The reported scores measure agreement with these heuristic labels and are provisional. Student-test results informed development and model selection; only the evaluator partition is reserved for the live evaluation.
