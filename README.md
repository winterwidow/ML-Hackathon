# PathSnap: GPS Trajectory Map Matching

PathSnap scores nearby directed road segments for noisy GPS observations and uses an HMM/Viterbi decoder to reconstruct a connected route. A FastAPI server serves predictions to the browser interface in `api/static/index.html`.

## Team Members

| Name | SRN |
| --- | --- |
| Tanvi Magalur | PES2UG24CS257 |
| Malepati Viswajith | PES2UG24CS259 |
| Muthuraj SM | PES2UG24CS289 |
| Naija Hodiegre | PES2UG24CS296 |

## Final model and results

The current final artifact is a Bayesian-tuned Extra Trees classifier (`160` trees, depth `16`, minimum leaf size `10`, `max_features=sqrt`, balanced class weights) followed by HMM/Viterbi decoding. Model selection prioritizes end-to-end route reconstruction. Hist Gradient Boosting achieved the strongest candidate macro F1 and log loss, while tuned Extra Trees achieved the highest HMM route scores in the permitted student-test development comparison. The full comparison and caveats are in [FINAL_MODEL_DECISION.md](FINAL_MODEL_DECISION.md).

| Measure | Result |
| --- | ---: |
| Five-fold grouped CV Top-1, Logistic Regression | 75.63% +/- 2.87 percentage points |
| Five-fold grouped CV Top-1, untuned Extra Trees | 75.61% +/- 2.61 percentage points |
| Tuned Extra Trees, 3-fold grouped training-only search | 75.89% +/- 2.76 percentage points |
| Student-test HMM weighted Top-1, tuned Extra Trees | 82.20% (classifier-only: 76.03%) |
| Student-test HMM mean Top-1 by trip, tuned Extra Trees | 85.19% |
| Student-test HMM weighted Top-1, Logistic Regression | 81.55% |
| Student-test candidate macro F1 / log loss, Hist Gradient Boosting | 0.8136 / 0.2340 (best of compared models) |

Student-test results are for development and informed model selection; they are not an untouched final benchmark. All labels are heuristic alignments. The evaluator partition was not used for fitting, tuning, or model selection. The model and aggregate result summaries are included; raw data, evaluator labels, large feature tables, per-observation predictions, and coordinate-bearing Folium maps remain local.

## Data placement

Keep the supplied dataset local under `data/`. Each partition contains one folder per trip with `.track`, `.nodes`, `.arcs`, and `.route` files:

```text
data/
  train/
  student_test/
  evaluator_test/
```

## Install

Use Python 3.10 or newer:

```powershell
python -m pip install -r requirements.txt
```

## Reproduce the pipeline

Run from the repository root. Candidate tables are large and ignored by Git; the commands below rebuild them locally.

```powershell
python pathsnap_person1.py ".\data" ".\features" train train_candidates_aligned.csv
python pathsnap_person1.py ".\data" ".\features" student_test student_test_candidates_aligned.csv
python cross_validate_models.py
python bayesian_tune_extratrees.py
python compare_student_models.py
python train_final_model.py
python hmm_decode.py
python plot_results.py
python build_report.py
```

The classifier uses seven inference-time features: time delta, GPS speed, candidate distance, road-segment length, GPS movement bearing, road bearing, and bearing difference. Route IDs, route-alignment distances, and reference labels are excluded from model inputs. The comparison includes Logistic Regression, Decision Tree, Random Forest, Extra Trees, and Hist Gradient Boosting, plus the tuned Extra Trees configuration. Cross-validation and Bayesian tuning keep trips grouped. The evaluator partition is never part of these commands.

Use files ending in `_aligned` under `features/` as the current summaries. Files without that suffix in the initial repository are superseded drafts from before the route-alignment correction.

## Folium map

After `hmm_decode.py`, render a local evaluation map for a student-test trip:

```powershell
python render_trip_map.py --trip-id TRIP_ID
```

The layer control toggles the GPS trace, nearby road network, reference route, and HMM-predicted route. These local HTML files contain trip coordinates and are ignored by Git. The OpenStreetMap tile background needs internet when the map is opened.

Create three training EDA maps for short, median, and long trips, each with GPS, road-network, and reference-route layers:

```powershell
python create_eda_folium_maps.py --data-root .\data
```

The HTML maps are written to `reports/maps/` and stay local.

## Two-laptop demo

On Laptop 1, place the evaluator partition under `data/evaluator_test/`, then start the service:

```powershell
$env:PATHSNAP_PARTITION = "evaluator_test"
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

On Laptop 2, open `http://LAPTOP1-IP:8000/`, set the API address to Laptop 1's IP, choose a trip, and request a route. Both laptops must share a network. The API reads reference routes server-side only after inference and returns no reference labels to the UI.

## Team responsibilities

See [docs/TEAM_DEMO_GUIDE.md](docs/TEAM_DEMO_GUIDE.md) for the role-by-role walkthrough, data and CSV explanations, model rationale, chart guide, limitations, judge Q&A, and two-laptop demo checklist. Generate the train/student-test EDA charts with `python eda.py --data-root .\data --output-dir .\eda`, and the aggregate model/HMM charts with `python plot_results.py`.

- **Person 1 - Malepati Viswajith - Data and features:** `pathsnap_person1.py`, `eda.py`, `create_eda_folium_maps.py`, alignment review, and data quality.
- **Person 2 - Naija Hodiegre - Models:** five-model comparison, grouped CV, Bayesian tuning, and final model training.
- **Person 3 - Tanvi Magalur - Route decoding:** HMM/Viterbi logic in `hmm_decode.py`.
- **Person 4 - Muthuraj SM - Demo integration:** `api/app.py`, browser UI, and two-laptop setup.

## Label limitations

Route labels are generated by a monotone spatial-alignment heuristic with a 50 m maximum labeling distance. Some training trips have large unlabeled portions. Scores measure agreement with these heuristic labels, not independently verified ground truth; describe them as provisional.
