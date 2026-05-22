# CS611 MLE Group Project — Context Handoff v4

*Last updated: 18 May 2026*
*Supersedes v3. Key changes: NEA weather dropped → MSS Weather substituted; two-speed business framing locked in; one-model-two-paths architecture finalised; proposal deck v2 rebuilt around new framing.*

---

## 1. Course & Assessment Context

- **Course:** CS611 — Machine Learning Engineering, SMU SCIS
- **Lecturer:** Ulysses Chong — Lead Data Scientist at Grab, Adjunct at SMU
- **Key rubric signal:** *"Don't focus on model complexity. Focus on how you solve a complex problem... explain design choices, that's where the marks are."*

### Deliverables

| # | Deliverable | Weight | Status |
|---|---|---|---|
| 1 | Proposal deck (5 min) | 20% | Week 6 — **rebuilt v2, ready for review** |
| 2 | Code + Report (max 8 pages, A4, Arial 11) | 30% | Week 10 — end of term |
| 3 | Final presentation (15 min) | 30% | Week 10 — end of term |
| 4 | Peer evaluation | 20% | End of term |

---

## 2. Project — Dengue Outbreak Risk Prediction

**Business problem (reframed v4):** Singapore's dengue control is a two-speed problem with two distinct operational loops:

- **Strategic loop:** Where should NEA fogging crews be deployed next week to suppress cluster *formation*? Weekly batch cadence, fogging-logistics action.
- **Tactical loop:** When a confirmed case is notified, which vulnerable residents in the immediate radius need same-day clinical outreach to prevent *severe dengue progression*? Event-driven cadence, CDC officer triage action.

**One-line pitch:** *"One model predicts subzone cluster risk. Two inference paths serve two different operational decisions — weekly fogging deployment and per-case clinical triage — with cost-asymmetric thresholds."*

---

## 3. Architecture — One Model, Two Inference Paths

The architectural decision locked in v4: **one trained ML model serves two inference applications.** Same MLflow artifact, two feature stores, two decision contexts.

### The single trained model

**Cluster Formation Model** — predicts `P(active dengue cluster in subzone S over the next 14 days)`. One training pipeline, one MLflow registry entry. Features: weather lags, cluster proximity, prior cluster history, seasonality, population density, vulnerability index, current case count.

### Inference path 1 — Strategic (weekly batch)

- **Trigger:** Airflow DAG, every Monday 06:00 SGT
- **Features pulled from:** Offline feature store (Postgres Gold layer / parquet) — stable, week-old data is fine
- **Process:** Score all 330 subzones in a single batch run
- **Output:** Risk tier (Low / Med / High) per subzone for the coming 14 days, written to `risk_tier` table
- **Consumer:** NEA Ops dashboard + Pest Control operators
- **Action:** Plan fogging crews, equipment staging, deployment routes for the week
- **Latency tolerance:** Days
- **Cost of false negative:** A missed fogging run — recoverable next week

### Inference path 2 — Tactical (event-driven real-time)

- **Trigger:** New row inserted into `confirmed_cases` table → Postgres LISTEN/NOTIFY fires
- **Pipeline:** Python listener catches notification → calls FastAPI `/predict` endpoint (in Docker)
- **Features pulled from:** Online feature store (Redis) — sub-ms row reads for `current_week_case_count` and `vulnerability_index`
- **Inference:** Same MLflow artifact loaded in memory, scores only the affected subzone
- **Decision layer:** Rules-based alert logic — `alert = (model_score > τ) AND (vulnerability_index > v_cut) AND (case_count >= n_min)`. **This is a transparent business rule, not a second ML model.**
- **Output:** Row written to `vulnerability_alerts` table
- **Consumer:** CDC officer dashboard at MOH
- **Action:** Triage decision — home visit, advisory push, testing van dispatch
- **Latency tolerance:** Sub-second (end-to-end target < 500 ms)
- **Cost of false negative:** Vulnerable resident enters WHO days 4–7 critical phase without surveillance — not recoverable

### Why one model, not two

We considered training a second model for the real-time path (e.g., predicting individual severity outcomes) and rejected it for three reasons:

1. **No labels exist** for individual-level dengue severity outcomes in public open data
2. **Fabricating labels** would amount to overfitting to invented ground truth
3. **The right engineering choice** is to apply the same predictive model's score inside a rules-based alerting layer that combines vulnerability and freshness deterministically

This is more defensible than hand-waving a second model into existence. "I picked the right number of models for the data I have" beats "I built two models because real-time is hard."

### Defending this to Ulysses

| Likely challenge | Response |
|---|---|
| "Why two layers if it's one model?" | Different features arrive at different speeds; different decisions have different cadences. Batch path uses weekly-stable features for weekly logistics. Real-time path uses event-driven features for case-by-case triage. Same predictive logic, different operational loops. |
| "Why not just read the batch score in real-time?" | The batch score can be up to 6 days stale, and the trigger event (a new case) changes the score's input. Reading stale scores means the officer is acting on a number that doesn't reflect the case in front of them. |
| "Why is the real-time alert layer not a separate model?" | No labelled outcome data for individual-level dengue severity exists publicly. The right engineering choice is to apply the same model's score inside a transparent rules-based alerting layer. Adding a second model without ground truth would be overfitting to invented labels. |

---

## 4. Data Sources (v4)

| Source | Role | Status |
|---|---|---|
| SGCharts cluster CSV | Anchor (labels + cluster history) | ⬜ To download |
| **MSS Weather Station Records** | Weather features 2013–present (replaces NEA) | ⬜ To consolidate from `station-records.xlsx` |
| SingStat population + age + comorbidity proxies | Demographics + vulnerability index | ⬜ To download (uploaded files present) |
| URA subzone GeoJSON | Spatial join reference | ⬜ To download (Master Plan 2019) |

### Sources removed since v3

- **LTA MRT tap volumes** — dropped (was already removed in v3)
- **NEA Weather API** — dropped. Rationale: API only returns data from May 2016 (temp/humidity) and Dec 2016 (rainfall); 7-hour ingest runtime impractical; partial data ingested is incomplete. MSS station records extend the usable training window back to 2013 — a strict improvement.

### Real-time event source (simulated)

- Table: `confirmed_cases` (Postgres)
- Trigger: ON INSERT → Postgres LISTEN/NOTIFY emits event on `case_event` channel
- In production this would be MOH's communicable disease notification feed; for the demo, a manual insert simulator generates events

### Uploaded files (current state)

| File | Purpose |
|---|---|
| `nea_weather.csv` | Partial NEA ingest — **deprecated**, will not be used |
| `station-records.xlsx` | MSS station data — **primary weather source** |
| `sgcharts_dengue.csv` | SGCharts dengue cluster data — anchor |
| `subzone_population.csv` | SingStat subzone population |
| `planning_area_population.csv` | SingStat planning area population |
| `dengue_proposal_v2.pptx` | Rebuilt proposal deck (v2) |

---

## 5. Feature Engineering Plan

All features engineered at Gold layer, per subzone per observation week. Same feature set across both inference paths; different feature stores serve them at different freshness.

| Feature | Rationale | Used by |
|---|---|---|
| Rainfall lag 1 week | Mosquito eggs laid last week hatch this week | Both paths |
| Rainfall lag 2 weeks | Peak breeding signal — Aedes cycle is 7–14 days | Both paths |
| Rainfall lag 4 weeks | Extended wet season effect | Both paths |
| Temperature rolling mean (4-week) | Higher temp accelerates Aedes lifecycle | Both paths |
| Relative humidity (7-day mean) | High humidity extends adult mosquito lifespan | Both paths |
| Cluster proximity score | Binary: any cluster within radius in prior 4 weeks | Both paths |
| Population density weight | SingStat residents per subzone, normalised 0–1 | Both paths |
| Seasonality (week-of-year sin/cos) | Singapore has two dengue peaks: May–Jun, Oct–Nov | Both paths |
| Prior cluster history (12-month) | Weeks with active cluster in past year — structural risk proxy | Both paths |
| Vulnerability index | % elderly + % with comorbidities per subzone, SingStat | Both paths (precomputed, stable) |
| Current week case count | Live case count from `confirmed_cases` | Both paths — *online store for tactical, offline for strategic* |

**Label:** Binary — does subzone S have an active dengue cluster in the 14 days after observation date T? Derived by spatial join of SGCharts cluster lat/lng to URA subzone boundaries using GeoPandas. Not synthetic — every label traces to a real cluster.

---

## 6. Key Design Decisions (v4)

| Decision | Choice | Rationale |
|---|---|---|
| Inference architecture | **One model, two paths** | Same trained artifact serves batch (Gold features, weekly) and real-time (Redis features, event-driven). Decision logic differs, not the predictive logic. |
| Real-time decision layer | **Rules-based, not a 2nd ML model** | No outcome-level labels exist publicly. `alert = score × vulnerability × case_count > threshold` is transparent and honest. |
| Inference cadence (strategic) | Weekly batch | Fogging schedules planned weekly; real-time latency adds no operational value to this decision |
| Inference cadence (tactical) | Event-driven real-time | CDC officer manages case-by-case triage during a working shift; sub-second latency is human reaction speed |
| Model | XGBoost + LightGBM challenger | Tabular + spatial data; interpretable via SHAP; trains in minutes locally |
| Cross-validation | Time-series CV (sliding window) | Standard k-fold leaks future cluster data — temporal leakage produces falsely optimistic AUC |
| Optimisation metric | Recall ≥ 0.70 (High-risk tier) | False negatives carry higher public-health cost than false positives |
| Subzone granularity | 330 subzones | Fine-grained enough for operational fogging deployment; coarser (55 planning areas) loses precision |
| Labels | Spatial join → binary | Only feasible approach given no direct subzone-level label exists publicly; not synthetic |
| Training window | 2013–2020 | Use full SGCharts window. 2020 serotype shift (DENV-3, 35k+ cases) demonstrates concept drift |
| Weather source | **MSS station records, not NEA API** | MSS extends usable window back to 2013; NEA has Dec 2016 floor + 7-hr ingest runtime |
| Deployment infra | Local Docker only | Prof explicitly said AWS spend ≠ better grade |
| Medallion Architecture | Bronze / Silver / Gold | Explicitly taught in course; demonstrates data engineering maturity |
| Feature store split | Offline (Postgres Gold + Parquet) + Online (Redis) | Same logical features; different freshness contracts |

---

## 7. Scripts

| Script | Purpose | Status |
|---|---|---|
| `ingest_nea_weather.py` | NEA weather API 2013–2020 | **DEPRECATED — replaced by MSS pipeline** |
| `ingest_mss_weather.py` | MSS `station-records.xlsx` → silver/gold parquet | ⬜ To write |
| `ingest_sgcharts.py` | Concatenate 250+ SGCharts CSVs into one time-series DataFrame | ⬜ To write |
| `preprocess.py` | Bronze → Silver: clean, validate, join on date + subzone | ⬜ To write |
| `feature_engineering.py` | Silver → Gold: lag features, spatial join, label creation | ⬜ To write |
| `train.py` | XGBoost/LightGBM, Optuna, MLflow tracking | ⬜ To write |
| `inference_batch.py` | Weekly batch scoring across 330 subzones | ⬜ To write |
| `inference_realtime.py` | FastAPI `/predict` endpoint, Postgres LISTEN/NOTIFY listener | ⬜ To write |
| `alert_rule.py` | Rules-based decision layer (score × vuln × count → ALERT/NO ALERT) | ⬜ To write |
| `monitor.py` | Drift detection, retraining trigger | ⬜ To write |
| `docker-compose.yml` | Postgres + Redis + Airflow + MLflow + FastAPI services | ⬜ To write |
| `build_deck.py` | Generate `dengue_proposal_v2.pptx` | ✅ Done |

---

## 8. Immediate Next Steps

1. **Audit `station-records.xlsx`** — confirm MSS coverage spans 2013–present and includes rainfall, temperature, humidity at station resolution
2. **Download SGCharts cluster data** — `outbreak.sgcharts.com/data` → unzip to `data/dengue/sgcharts/`
3. **Download URA subzone GeoJSON** — Master Plan 2019 from `data.gov.sg`
4. **Confirm SingStat data** — verify `subzone_population.csv` includes age breakdowns needed for vulnerability index
5. **Write `ingest_mss_weather.py`** — consolidate station-records into a clean daily-station-keyed parquet
6. **Write `ingest_sgcharts.py`** — concatenate 250+ snapshots into a time-keyed DataFrame
7. **Write `preprocess.py`** — Bronze → Silver
8. **Write `feature_engineering.py`** — Silver → Gold + spatial join via GeoPandas
9. **Set up `docker-compose.yml`** — local Postgres + Redis + MLflow stack

---

## 9. Python Environment

- Python 3.14 at `/Users/garethwang/Library/Python/3.14/`
- Installed: `requests`, `pandas`, `python-pptx`
- Still needed: `geopandas`, `shapely`, `pyspark`, `scikit-learn`, `xgboost`, `lightgbm`, `mlflow`, `optuna`, `fastapi`, `uvicorn`, `redis`, `psycopg2`, `openpyxl`
- Long scripts: `caffeinate -is python3 ~/Desktop/script_name.py`

---

## 10. Proposal Deck v2 — Slide Map (5-min talk)

The rebuilt deck (`dengue_proposal_v2.pptx`) has 7 slides, ~40 seconds each. Speaking points below mirror the slide content but in delivery order.

### Slide 1 — Title
*Open with the project name and the one-line architectural claim.*
> "Dengue Outbreak Risk Prediction. We're predicting subzone-level dengue cluster risk using one model that serves two different operational decisions — weekly fogging deployment and real-time clinical triage."

### Slide 2 — Two-speed problem (~50s, anchor slide)
*Establish the framing that the rest of the deck depends on.*
> "Singapore's 2020 outbreak — 35,000 cases — exposed two operational gaps, not one. NEA needs to know where to fog next week — that's a strategic, weekly decision. MOH's CDC officers need to know who to flag right now when a confirmed case is notified — that's a tactical, sub-second decision. Conflating these with one inference architecture is the strategic mistake. Two different cost functions justify two different inference paths, served by one underlying model."

### Slide 3 — Dataset (~40s)
*Walk left-to-right: anchor, weather, demographics+geometry, label. Mention real-time event stream.*
> "SGCharts cluster snapshots are our anchor — 7 years of real spatial data including the 2020 outbreak. MSS station records give us weather features back to 2013. SingStat plus URA give us subzone demographics and geometry for spatial joins. Labels are derived by intersecting SGCharts cluster polygons with URA subzones — every label traces to a real cluster, none synthetic. The real-time path is fed by a simulated `confirmed_cases` event stream — a Postgres LISTEN/NOTIFY trigger."

### Slide 4 — Users (~30s)
*Don't read every card. Anchor on the contrast.*
> "Four users across two loops. NEA Ops and Pest Control are strategic-loop users — they consume the weekly risk map. CDC officers at MOH are tactical-loop users — they receive per-case alerts in real time. Internal data ops monitor both."

### Slide 5 — Architecture macro (~45s)
*This is the story slide. Trace inputs → model → two outputs.*
> "Left: what we know. Middle: what we predict — one Cluster Formation Model giving probability of cluster formation in the next 14 days. Right: what we do — two different actions served by the same model. The strategic loop produces a weekly risk map for fogging deployment. The tactical loop produces per-case alerts for clinical triage. Same predictive logic, different freshness contracts, different thresholds, different downstream actions."

### Slide 6 — Architecture detail (~50s)
*This is the proof slide. Reference batch lane briefly, real-time chain in detail.*
> "Same diagram with the bolts and nuts. Process Data column: medallion architecture with both offline feature store on Postgres Gold and online feature store on Redis. Develop Model: one MLflow registry, one trained artifact. Deploy splits into two lanes — top lane is the weekly batch path, Airflow → score 330 subzones → risk_tier table → NEA dashboard. Bottom lane is the real-time path: a confirmed case insert fires Postgres LISTEN/NOTIFY, a Python listener calls our FastAPI endpoint in Docker, online features come from Redis, the same MLflow artifact scores the subzone, our rules-based alert layer decides ALERT or NO ALERT, and the result is written for the CDC officer. End-to-end target under 500ms. All containerised via Docker compose, no cloud spend."

### Slide 7 — Design choices & decisions (~45s)
*Lead with the two-layer choice; emphasize that the right column resolves what was open at proposal stage.*
> "Four design choices, four decisions resolved since we started. The big choice: two layers by design, because the operational loops are genuinely different. The big resolution: the real-time alert layer is a rule, not a second model — we'd be inventing labels otherwise. Weather source switched from NEA to MSS to extend the training window. Training window locked at 2013–2020 with 2020 as our concept drift demonstration."

---

## 11. Q&A Prep — Likely Ulysses Challenges

| Challenge | Concise answer |
|---|---|
| "Why a 14-day prediction window?" | Aedes aegypti egg-to-adult cycle is 7–14 days. Fogging deployed within this window suppresses cluster formation before transmission peaks. |
| "Why subzones, not planning areas?" | Fogging is deployed at street/block level. 330 subzones is the smallest unit aligned with operational deployment; 55 planning areas loses deployment precision. |
| "Why two layers if it's one model?" | Different features arrive at different speeds; different decisions have different cadences. Batch path operates on weekly-stable features for weekly logistics. Real-time path operates on event-driven features for case-by-case triage. Same predictive logic, different operational loops. |
| "Why not just read the batch score in real-time?" | The batch score can be 6 days stale, and the new case event itself changes a key input feature (case count). Reading the stale score means acting on a number that doesn't reflect current state. |
| "Why is the real-time alert layer not a 2nd model?" | No labelled outcome data exists publicly for individual-level dengue severity. Score × vulnerability × case count is a transparent rule. A second model without ground truth would be overfitting to invented labels. |
| "How do you handle the 2020 serotype shift?" | Train on 2013–2020 inclusive. Treat 2020 as a documented concept drift event in the report — show feature drift metrics around the DENV-3 emergence. |
| "Where does ground truth come from for evaluation?" | SGCharts cluster snapshots are our ground truth for the binary label. Time-series CV with sliding window prevents temporal leakage. |
| "What if the real-time path goes down?" | The strategic batch path is unaffected — it runs independently weekly. Real-time degrades gracefully: CDC officers fall back to the last batch risk tier for the subzone. No cascading failure. |
| "Why XGBoost over deep learning?" | Tabular spatial + temporal data with ~50 features. XGBoost and LightGBM train in minutes locally, are interpretable via SHAP for public health reporting, and outperform NN baselines on this data regime. Deep learning has no advantage here. |
| "Why Recall ≥ 0.70 not F1?" | Asymmetric public-health cost: a missed outbreak in High tier means people get sick. We optimise for the more expensive error. F1 treats precision and recall as symmetric — they aren't. |
| "How does the model update when MOH adds a new case mid-week?" | Real-time path: the new case appears in `current_week_case_count` in Redis immediately, so the next real-time scoring request for that subzone uses the fresh count. Batch path: incorporates the new case at next Monday's run. |
| "Why MSS instead of NEA?" | MSS station records extend the usable training window back to 2013. NEA's realtime API only returns data from May/Dec 2016 onwards, with a 7-hour ingest runtime and frequent gaps. MSS is a strict improvement. |
| "What's the cost of running this in production?" | Local Docker for the project — zero cloud spend. In production: one Postgres instance, one Redis cluster for online features, MLflow registry, FastAPI container — all standard infrastructure NEA would already have. |

---

## 12. Outstanding Items

- Confirm MSS dataset structure once `station-records.xlsx` is audited
- Verify URA Master Plan 2019 subzone count matches the 330 figure used throughout the deck
- Decide on real-time alert thresholds — initial defaults can be set from training data quantiles
- Finalise CDC officer dashboard mock — Streamlit or simple HTML reading from `vulnerability_alerts`
- Choose `confirmed_cases` simulator approach — Faker-generated stream or replay of historical case data
