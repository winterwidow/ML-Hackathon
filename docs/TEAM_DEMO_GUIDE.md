# PathSnap Team and Demo Guide

This is the shared briefing for the four presenters. It explains the problem, the data, the current implementation, the results, the charts, and the two-laptop demo. It describes what the submitted files actually do; it does not treat every item in the role plan as proof that a separate artifact was produced.

## 1. Project in one minute

GPS devices report noisy positions. A point can fall between roads or near several parallel roads, so selecting the nearest road independently at every timestamp can create an implausible zig-zag. PathSnap:

1. projects the trip and road graph into local metre coordinates;
2. finds nearby directed road segments for each GPS point;
3. scores each point/road candidate with a trained binary classifier;
4. uses an HMM-style Viterbi decoder to find a sequence that balances those scores with directed road connectivity; and
5. serves the route to a browser map through FastAPI.

```mermaid
flowchart LR
  A[Noisy GPS points] --> B[Local metre projection]
  R[Road nodes and directed arcs] --> B
  B --> C[Nearby candidate arcs]
  C --> D[Seven-feature Extra Trees candidate scorer]
  D --> E[HMM / Viterbi route decoder]
  E --> F[FastAPI]
  F --> G[Browser map: GPS and predicted route]
```

**Short presentation line:** “The classifier scores plausible roads at each GPS point; Viterbi uses road direction and continuity to turn those scores into one coherent route.”

## 2. What the supplied files mean

Each trip folder has four whitespace-separated files, normally without a header:

| File | Meaning | How the code uses it |
| --- | --- | --- |
| `<trip>.track` | Longitude, latitude, timestamp for each GPS observation | The noisy input sequence; the timestamp differences are treated as seconds. |
| `<trip>.nodes` | Longitude, latitude for each road node | Coordinates used to build road segments. |
| `<trip>.arcs` | Start-node index and end-node index for each directed road segment | Arc IDs are the row positions in this file; direction matters. |
| `<trip>.route` | One directed arc ID per reference route step | Used to create heuristic labels for development and to score the evaluator trip on the server after prediction. |

Node and arc indexes are zero-based and belong to their trip. Arc ID 12 in one trip is unrelated to arc ID 12 in another. The `.route` file is reference information: it is not an inference feature.

### Partitions and data handling

The provided split is 80 training trips, 10 `student_test` trips, and 10 `evaluator_test` trips. The implementation keeps trips intact when making group-aware validation folds.

- `train`: fit the model and run grouped model selection/tuning.
- `student_test`: development evaluation after selection; useful for the report, but not an untouched final benchmark.
- `evaluator_test`: reserved for the live evaluator demonstration. Do not use it to tune features, choose a model, or adjust parameters.

The API runs inference first. It reads a reference route on the server only after inference to calculate an aggregate score. It does not send reference arc IDs to the browser.

## 3. Person-by-person responsibilities and handoff

### Person 1 — Data, EDA, and features

**What this role owns:** trip/data checks, coordinate projection, candidate generation, label alignment, feature-table definition, candidate misses, EDA/report section 2.2, and Folium EDA maps if required by the judging rubric.

**Main files:** [`pathsnap_person1.py`](../pathsnap_person1.py), [`eda.py`](../eda.py), [`create_eda_folium_maps.py`](../create_eda_folium_maps.py), and generated files in `eda/`.

**What the current implementation does:**

- Reads each trip’s track, nodes, arcs, and route, and checks required indexes and timestamp order.
- Projects longitude/latitude into a local equirectangular plane measured in metres. This avoids calculating road distance directly in degrees. It is a practical local approximation, not a formal UTM reprojection.
- Indexes road-segment bounding boxes in a 100 m spatial grid.
- For each GPS point, considers up to 8 road candidates within 80 m.
- Aligns the GPS sequence monotonically to the reference route. A route label is kept only when the alignment distance is at most 50 m.
- Creates the seven model features described below, plus metadata and label-only columns.
- The current EDA script reads `train` and `student_test` only. It deliberately does not open `evaluator_test`.
- `create_eda_folium_maps.py` builds local short-, median-, and long-trip training maps with separate GPS, nearby-road-network, and reference-route layers. It writes coordinate-bearing HTML only under `reports/maps/`, which Git ignores.

**Important results to report:**

| Partition | Trips | GPS observations | Route-labeled observations | Positive route candidate present | Candidate misses |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 80 | 208,417 | 159,244 (76.40%) | 157,218 | 2,026 (1.27% of labeled observations) |
| Student test | 10 | 19,160 | 18,846 (98.36%) | 18,656 | 190 (1.01% of labeled observations) |

EDA found no invalid GPS coordinates, invalid timestamps, or invalid arc endpoints across these 90 development trips. It found zero isolated road nodes in these partitions. Some trips still have multiple **weakly connected components**: that metric ignores arc direction, so it describes disconnected pieces in the undirected sense, not whether every directed route is reachable from every other arc.

**How to explain a candidate miss:** “The correct route arc was not among the generated candidates. A classifier cannot select a road it was never offered.” Candidate coverage is therefore a separate ceiling on model performance.

**Handoff to Person 2:** the shared candidate table convention is one row per `(trip_id, obs_idx, candidate_arc_id)`. The feature names in the model artifact are the agreed classifier input order. Keep `aligned_route_arc_id`, `label_known`, `is_route_label`, and `route_alignment_dist_m` out of model inputs.

### Person 2 — Models and Bayesian search

**What this role owns:** training five model families, grouped evaluation, the chosen model, hyperparameter search, comparison plots, and report sections 3 and 4.

**Main files:** [`train_models.py`](../train_models.py), [`cross_validate_models.py`](../cross_validate_models.py), [`bayesian_tune_extratrees.py`](../bayesian_tune_extratrees.py), [`compare_student_models.py`](../compare_student_models.py), [`train_final_model.py`](../train_final_model.py), and [`plot_results.py`](../plot_results.py).

**The five models compared:**

| Model family | Why include it |
| --- | --- |
| Logistic Regression | Fast, interpretable baseline; outputs candidate probabilities. |
| Decision Tree | Simple nonlinear baseline that can learn feature thresholds. |
| Random Forest | Bagged trees reduce the variance of one decision tree. |
| Extra Trees | Randomized tree ensemble; useful comparison against Random Forest. |
| Histogram Gradient Boosting | Boosted trees can model nonlinear feature interactions and often give strong probabilities. |

The final artifact is **Extra Trees**, with `n_estimators=160`, `max_depth=16`, `min_samples_leaf=10`, `max_features="sqrt"`, and `class_weight="balanced"`. Each tree uses randomized feature/split choices; averaging trees provides a nonlinear candidate probability score without requiring feature scaling. The HMM/Viterbi decoder uses those probabilities plus directed-arc connectivity to reconstruct a sequence.

**How the model was selected:** five-fold `GroupKFold` keeps entire trips together. The final goal is route reconstruction, so we compared both candidate ranking and classifier-plus-HMM route scores. The allowed 10-trip student-test partition was used for development-stage comparison and informed the selection; it is therefore not an untouched final benchmark. The 10 evaluator trips were not used for model selection or tuning.

| Model | Observation Top-1, mean ± std | Candidate macro F1 | Candidate log loss |
| --- | ---: | ---: | ---: |
| Logistic Regression | 75.63% ± 2.87 pp | 0.7883 | 0.2628 |
| Extra Trees | 75.61% ± 2.61 pp | 0.7555 | 0.3627 |
| Hist Gradient Boosting | 73.94% ± 2.68 pp | **0.8113** | **0.2274** |
| Random Forest | 73.89% ± 2.85 pp | 0.7877 | 0.2944 |
| Decision Tree | 70.42% ± 3.40 pp | 0.7518 | 0.6472 |

**Bayesian tuning:** a 10-trial Gaussian-process expected-improvement search optimized observation Top-1 with three `GroupKFold` splits using training trips only. It selected 160 trees, depth 16, minimum leaf size 10, square-root feature sampling, and balanced class weights. The result was 75.89% ± 2.76 pp, compared with about 75.76% for untuned Extra Trees and 75.62% for Logistic Regression on the same three screening folds. The tuning gain is modest.

**Hist Gradient Boosting check:** this boosted-tree model was included in the five-family comparison and evaluated on the same student-test trips. It led the candidate-level metrics (macro F1 0.8136; log loss 0.2340), but its HMM weighted route Top-1 was 79.62%. Tuned Extra Trees had lower candidate F1 and worse log loss, but the highest HMM weighted route Top-1 (82.20%) and macro-by-trip score (85.19%) in this development comparison. Thus Extra Trees is selected for the route reconstruction objective; if candidate probability quality becomes the objective, Hist Gradient Boosting is a strong alternative.

| Student-test model | Candidate macro F1 | Candidate log loss | Classifier Top-1 | HMM weighted Top-1 | HMM mean by trip |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.7955 | 0.2706 | 76.33% | 81.55% | 84.45% |
| Decision Tree | 0.7474 | 0.6371 | 69.74% | 76.33% | 78.33% |
| Random Forest | 0.7816 | 0.3088 | 73.36% | 80.02% | 82.53% |
| Extra Trees, untuned | 0.7517 | 0.3747 | 75.83% | 81.68% | 84.80% |
| Hist Gradient Boosting | **0.8136** | **0.2340** | 73.73% | 79.62% | 82.21% |
| Extra Trees, Bayesian tuned | 0.7495 | 0.3832 | **76.03%** | **82.20%** | **85.19%** |

The tuned Extra Trees HMM result improved weighted Top-1 by 0.65 percentage points versus Logistic Regression + HMM (about 122 of the 18,846 heuristic route labels). It is a development result, not a guaranteed score on evaluator trips. The selected model's candidate confusion matrix at probability threshold 0.5 is TN=87,845, FP=18,296, FN=2,615, TP=16,041. Candidate classification at a threshold is distinct from choosing the highest-scoring candidate for each observation.

### Person 3 — HMM and Viterbi route reconstruction

**What this role owns:** candidate states, emissions, transitions, handling gaps/missing candidates, route continuity, comparison against the classifier-only path, map layers, and report sections 5 and 6.

**Main files:** [`hmm_decode.py`](../hmm_decode.py), [`render_trip_map.py`](../render_trip_map.py), and the selected-model comparison in `features/student_test_model_comparison_aligned.csv`.

**Decoder in plain language:**

- A hidden state at time `t` is one candidate directed road arc for GPS point `t`.
- The emission score is the log of the classifier probability for that candidate. A likely candidate receives a better score.
- Staying on the same arc has transition score 0. Moving to an arc whose start node equals the previous arc’s end node gets -1. A disconnected jump gets -5.
- Viterbi accumulates emission plus transition scores over the sequence and uses back-pointers to recover the highest-scoring whole path. It can select a slightly lower-scoring candidate at one point if doing so makes the whole sequence more plausible.
- Decoding restarts across a missing-candidate observation, a non-consecutive observation index, or a GPS time gap longer than 60 seconds. It does not force a route across an outage.

**Student-test aggregate result for the selected tuned Extra Trees model:** weighted all-label Top-1 increased from 76.03% (classifier only) to 82.20% (classifier + HMM). Mean Top-1 across the 10 trips increased from 78.80% to 85.19%. Mean continuity across decoded adjacent observations was 99.50%. The HMM did not improve every individual trip, and these fixed transition penalties were not separately nested-tuned.

**What the maps show:** the browser UI draws the GPS trace as dashed blue with a white outline, the predicted road route in red, and green/purple start/end markers. `render_trip_map.py` also provides toggleable local Folium layers for the GPS trace, nearby road network, reference route, and HMM-predicted route. `create_eda_folium_maps.py` builds training EDA maps with GPS, road-network, and reference-route layers. Both scripts save coordinate-bearing HTML under `reports/maps/`; keep those files local and do not upload them.

### Person 4 — API, UI, deployment, and integration

**What this role owns:** FastAPI endpoints, the trip selector and map UI, two-laptop networking, logging/screenshots, repo setup, README/requirements, final report assembly, and report section 7.

**Main files:** [`api/app.py`](../api/app.py), [`api/static/index.html`](../api/static/index.html), [`README.md`](../README.md), [`requirements.txt`](../requirements.txt), and [`build_report.py`](../build_report.py). The final report is `reports/PathSnap_Hackathon_Report.pdf`.

**Endpoints:**

| Request | Purpose |
| --- | --- |
| `GET /` | Serves the browser map interface. |
| `GET /health` | Confirms the API is running and shows active partition/model. |
| `GET /trips` | Returns available trip IDs for the server’s selected partition. |
| `POST /predict` with `{"trip_id":"..."}` | Runs candidate scoring and Viterbi, returns GPS and predicted-route coordinates plus an aggregate score when reference labels are available on the server. |

## 4. Features and code flow

### Seven classifier features

| Feature | Plain-language meaning |
| --- | --- |
| `dt_s` | Seconds since the previous GPS observation. |
| `gps_speed_mps` | Distance moved between GPS points divided by elapsed time. |
| `candidate_dist_m` | Perpendicular distance from GPS point to this road segment. |
| `road_length_m` | Length of the directed road segment. |
| `gps_move_bearing_rad` | Direction of observed GPS movement. |
| `road_bearing_rad` | Direction of the candidate segment. |
| `bearing_diff_rad` | Smallest wrapped angular difference between movement and road direction. |

`partition`, `trip_id`, `obs_idx`, arc IDs, GPS timestamp, alignment distance, and route labels are metadata or label/debug fields. They are not among the model inputs. Excluding route-derived labels and distances prevents reference information from leaking into inference.

### Key functions to point to in code

- `pathsnap_person1.py`: `project`, `build_segments`, `spatial_index`, `nearby`, `align_route_observations`, and `process_trip` implement the map-matching data preparation.
- `train_models.py`: defines the feature list, negative sampling, candidate scoring, and initial trip-grouped holdout comparison.
- `cross_validate_models.py`: defines all five estimators and the five-fold trip-grouped evaluation.
- `bayesian_tune_extratrees.py`: runs a 10-trial Bayesian search over Extra Trees settings with grouped folds from the training trips only.
- `train_final_model.py`: refits tuned Extra Trees on all training trips and writes `models/extra_trees_candidate_model.joblib`.
- `hmm_decode.py`: `decode` splits observations into continuous segments, scores transitions, then backtracks through Viterbi pointers.
- `api/app.py`: `infer` reconstructs candidates from raw trip files, calls `predict_proba`, invokes `decode`, then packages the predicted route for the UI.

The API tries the standard 80 m candidate radius first and, only if it finds none, retries within 500 m. That fallback can help avoid an empty display, but very distant fallback candidates are outside the main 80 m training candidate range and should be interpreted cautiously.

## 5. What each result CSV contains

Use files ending in `_aligned` for the finalized model results. Earlier files without that suffix are preserved first-run drafts and should not be mixed into the final comparison. Large candidate tables and row-level predictions are intentionally kept local.

### Public aggregate result files

| File | What each row means / how to use it |
| --- | --- |
| `features/data_quality_summary.csv` | One row per partition; trip/observation totals, route-label coverage, positive candidate coverage, and candidate misses. |
| `features/initial_model_comparison_aligned.csv` | One row per model from the earlier 80/20 trip-grouped holdout experiment. It is a preliminary check; use the five-fold file below for final model choice. |
| `features/group_cv_folds_aligned.csv` | One row per model per validation fold, including candidate accuracy, macro F1, log loss, observation Top-1, training seconds, and inference time. |
| `features/group_cv_summary_aligned.csv` | One row per model, summarizing the five grouped folds with mean and standard deviation. This is the main comparison table. |
| `features/group_cv_protocol_aligned.txt` | States the grouping, fold count, negative sampling, and which partitions were excluded. |
| `features/bayesian_trials_aligned.csv`, `features/bayesian_best_aligned.txt`, and `features/bayesian_comparison*.csv` | Earlier Logistic Regression `C` search artifacts. Keep them as baseline experiments; they do not define the selected model. |
| `features/extra_trees_bayesian_trials_aligned.csv` | One row per Extra Trees search trial; validation folds contain training trips only. |
| `features/extra_trees_bayesian_best_aligned.txt` | Best Extra Trees configuration and grouped-CV objective score. |
| `features/student_test_model_comparison_aligned.csv` | Aggregate comparison of five model families plus tuned Extra Trees on the 10 student-test trips, including candidate, HMM route, continuity, and timing metrics. |
| `features/student_test_metrics_aligned.csv` | One-row evaluation of the selected tuned Extra Trees model, including all-label Top-1, candidate metrics, training/inference time, and misses. |
| `features/student_test_hmm_aggregate.csv` | Selected-model classifier-vs-HMM aggregates for weighted Top-1, per-trip macro Top-1, and continuity, with metric definitions. |

### Local files; do not publish them

| File | Why it stays local |
| --- | --- |
| `features/train_candidates_aligned.csv` and `features/student_test_candidates_aligned.csv` | Large row-level tables with trip/observation/arc identifiers and route-derived labels. Regenerate from local data. |
| `features/student_test_hmm_predictions_aligned.csv` | Per-observation predicted and reference arc IDs; this is not an aggregate result. |
| `features/student_test_hmm_summary_aligned.csv` | Per-trip scores keyed by trip ID; keep with local evaluation artifacts. |
| `eda/eda_trip_summary.csv` | Per-trip data-quality statistics. The aggregate partition summary can be shared instead. |
| Folium HTML maps | Contain GPS route coordinates. Keep on the machine used for local inspection. |

`eda/eda_partition_summary.csv` is the aggregate output from the current EDA run. `features/README.md` describes the aligned-file convention and local-only large files.

## 6. EDA and model charts: what they show

The current local EDA output is in `eda/`. It was generated from the 80 train and 10 student-test trips only. To regenerate it from the project root, run `python eda.py --data-root .\data --output-dir .\eda`. To regenerate the model/HMM charts, run `python plot_results.py`.

| Chart | What is plotted | How to explain the result |
| --- | --- | --- |
| `01_trip_structure.png` | Per-trip GPS, node, directed-arc, and route-arc counts; log y-axis; outliers hidden for readability. | Shows trip/network sizes vary substantially. The log scale lets small and large trips appear together. |
| `02_observations_per_trip.png` | Histogram of GPS observations per trip by partition. | Shows how much trip length/sequence size varies. Large trips matter for runtime and weighted metrics. |
| `03_motion_distributions.png` | GPS-derived speed distribution (0–60 m/s displayed) and trip duration distribution. | Helps spot implausible speeds and very short/long trips. The displayed speed range is clipped for readability, not a claim that no higher values exist. |
| `04_sampling_gaps.png` | GPS observation count against number of timestamp gaps over one second, one point per trip. | Shows which trips have more sampling interruptions; it does not show gap durations themselves. |
| `05_network_quality.png` | Weakly connected component count against isolated-node count, one point per trip; symlog axes. | Isolated nodes are zero in these development trips, while component counts vary. Weak connectivity ignores one-way direction. |
| `06_candidate_coverage.png` | Route-label coverage and fraction of labeled observations without a positive road candidate by partition. | Candidate misses happen before classification, so they are a preprocessing/coverage issue as well as a limit on the final score. |
| `07_representative_trajectories.png` | Six training GPS traces centered at their own start point, axes in metres. | Illustrates trip shapes without raw longitude/latitude axes. These are still trajectory-derived visuals; keep the file local. |
| `08_model_comparison.png` | Five-fold grouped mean Top-1, candidate macro F1, and log loss; error bars show fold standard deviation. Blue marks the Extra Trees family selected after tuning; gray bars are alternatives. | Logistic Regression and untuned Extra Trees are nearly tied on grouped Top-1; Hist Gradient Boosting leads candidate F1 and log loss. |
| `09_hmm_effect.png` | Tuned Extra Trees classifier-only vs HMM weighted Top-1, macro-by-trip Top-1, and continuity. | On the student-test development trips, the HMM improves route agreement and continuity; the aggregate result does not mean every trip improved. |
| `10_student_model_routes.png` | Classifier-only and HMM weighted Top-1 for all six student-test model variants. | Tuned Extra Trees + HMM has the highest route score in this development comparison. |
| `11_student_candidate_metrics.png` | Candidate macro F1 and log loss across the six student-test model variants. | Hist Gradient Boosting leads candidate classification and probability quality, which are distinct from final route scores. |

**Metric vocabulary:**

- **Candidate accuracy/F1/log loss** evaluate each candidate row as positive route arc or negative candidate. Macro F1 weights the positive and negative class F1 scores equally. Lower log loss is better and rewards calibrated probabilities.
- **Observation Top-1** groups candidates by GPS observation, chooses the candidate with highest probability, and checks if it is the aligned route arc. This matches the classifier’s ranking job.
- **Weighted score** pools all labeled GPS observations, so long trips count more. **Macro-by-trip score** calculates each trip separately and averages the ten trip scores, so each trip counts equally.
- **Continuity** is the fraction of decoded adjacent observation pairs that stay on the same directed arc or move to a directed successor. It is not route accuracy by itself; a wrong route can still be internally continuous.
- **Candidate miss rate** counts labeled GPS observations whose reference road is absent from the candidates. Such an observation cannot be fixed by changing the classifier.

## 7. Live two-laptop demo

Your earlier demo run showed `status: ok`, a populated trip selector, and the blue GPS trace with the red predicted path. Use this checklist on demo day.

### Laptop 1 — model and API

From the PathSnap project folder on Laptop 1:

```powershell
$env:PATHSNAP_PARTITION = "evaluator_test"
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Keep the evaluator data on Laptop 1 under `data\evaluator_test\`. `0.0.0.0` makes the server listen on the laptop’s network interfaces; it is not the address Laptop 2 should type. Confirm Laptop 1’s current IPv4 address on the same Wi-Fi/network. The previous address used was `10.0.15.101`; it may change.

### Laptop 2 — evaluator UI

1. Open `http://<Laptop-1-IPv4>:8000/` in a browser, replacing the placeholder with Laptop 1’s current network address.
2. Wait until the status says how many trips loaded.
3. Select one of the listed evaluator trip IDs and click **Reconstruct route**.
4. Explain the display: dashed blue with white outline = observed GPS; red = predicted road path; green = start; purple = finish. If the evaluator provides an accuracy score, it is an aggregate server-side comparison.

### If the dropdown or map fails

- On Laptop 1, open `http://127.0.0.1:8000/health`; confirm `status` is `ok` and `partition` is `evaluator_test`.
- On Laptop 2, use Laptop 1’s current IPv4, not `0.0.0.0`; leave the API-address field empty to use the page’s own server address.
- Confirm both laptops are on the same network and Laptop 1 allows Python/Uvicorn through the private-network firewall.
- If the selector works but map tiles are blank, the OpenStreetMap tile service may be unreachable. The route lines still come from the API; the background map needs internet access.
- If no road candidates are found, the map may show GPS but little/no red route. Try another listed trip and report it as a candidate-coverage limitation rather than claiming the route exists.

## 8. Suggested speaking order and judge questions

### 4-person presentation flow

1. **Person 1 (about 45 seconds):** explain the files and trip split, why coordinates are projected into metres, how nearby candidates are generated, the seven features, and the 1.01% student-test candidate miss rate.
2. **Person 2 (about 60 seconds):** show the five-family comparison, explain trip-grouped CV and the separate Bayesian search, then compare tuned Extra Trees + HMM route scores with Hist Gradient Boosting's better candidate F1/log loss. State that student-test results are development metrics.
3. **Person 3 (about 45 seconds):** explain candidate states, probability emissions, same/connected/disconnected transition scores, the 60-second gap break, and the HMM improvement with its limitations.
4. **Person 4 (about 60 seconds):** show `/health`, the trip selector, select a trip, reconstruct, identify the map colors, and explain the two-laptop setup and server-side reference labels.

The current repo does not include captured demo screenshots. Person 4 should save screenshots of `/health`, the populated trip selector, and the finished map during the live run if the submission form asks for them.

### Likely questions and prepared answers

**Why not choose the geometrically nearest road?** GPS noise, parallel roads, and intersections can make the nearest road wrong at one instant. The model uses distance and movement/road direction, and the HMM uses sequence continuity.

**Why Extra Trees if Hist Gradient Boosting has better F1 and log loss?** Candidate F1 and log loss score each candidate row. For the route task, we also compare the decoded route. Tuned Extra Trees had the highest student-test HMM weighted and macro-by-trip Top-1. Hist Gradient Boosting is better on candidate-level metrics. The student-test comparison informed development selection, so it is not an untouched benchmark.

**What did Bayesian tuning change?** It selected 160 trees, depth 16, minimum leaf size 10, square-root feature sampling, and balanced class weights. On three grouped training folds, Top-1 increased modestly from about 75.76% for untuned Extra Trees to 75.89%.

**What does a candidate miss mean?** The correct reference arc was not in the generated candidate list. No classifier can rank a missing candidate, so the miss is counted wrong in all-observation Top-1.

**Is 82.20% the same as the 85.19% HMM score?** No. 82.20% pools all labeled observations, giving long trips more weight. 85.19% averages per-trip accuracies, giving each of the 10 trips equal weight.

**Did we use the evaluator labels to train or tune?** No. Training and tuning use the 80 train trips. The 10 student-test trips are a development evaluation. The evaluator partition is selected only for the live API; the route reference is read after inference on Laptop 1 for aggregate scoring.

**Are the reported labels perfect ground truth?** No. The `.route` labels are aligned to GPS with a monotone spatial heuristic and 50 m threshold. All scores measure agreement with those heuristic labels and are provisional; long GPS gaps, intersections, and repeated roads can affect alignment.

**Why can candidate macro F1 differ from observation Top-1?** Candidate F1 decides positive/negative separately for every candidate row at a threshold. Observation Top-1 ranks candidates against one another within each GPS point. They answer different questions.

## 9. Limits to state clearly

- The route labels are heuristic, not independently verified ground truth.
- Candidate generation limits what the classifier can recover; student-test misses are 190 of 18,846 labeled observations.
- The local equirectangular projection is metre-based and suitable as a local approximation, but is not a formal projected CRS implementation.
- HMM transition penalties are a simple fixed baseline, not separately nested-tuned; one trip can get worse even when the aggregate improves.
- Student-test results are a development check; do not market them as an untouched final benchmark.
- Inference timing depends on the machine and is not a guaranteed latency.

## 10. File quick index

- **Report:** `reports/PathSnap_Hackathon_Report.pdf`
- **Final model:** `models/extra_trees_candidate_model.joblib`
- **Role 1:** `pathsnap_person1.py`, `eda.py`, `eda/`
- **Role 2:** `train_models.py`, `cross_validate_models.py`, `bayesian_tune_extratrees.py`, `compare_student_models.py`, `train_final_model.py`, `evaluate_student_test.py`, `plot_results.py`, `FINAL_MODEL_DECISION.md`
- **Role 3:** `hmm_decode.py`, `render_trip_map.py`
- **Role 4:** `api/app.py`, `api/static/index.html`, `README.md`, `requirements.txt`, `build_report.py`
- **Shared results:** `features/*_aligned*`; check the public/local tables in section 5 before sharing any file.
