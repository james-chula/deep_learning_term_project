# Report and Reproducibility Status

**Project:** Blood Cell Classification with ResNet-18  
**Owner:** Report and Reproducibility Lead  
**Last updated:** 2026-11-13 10:29 Asia/Bangkok (UTC+07:00)  
**Repository state (Project Lead):** main at `fded1f25216f00a0cdf09a46be6fe39c8c19dab8` (`docs: add project state and experiment protocol`), clean before this state update, tracking `origin/main`. Documentation only; no authorized execution manifest/config exists yet.  
**Git:** Project Lead only; this owner performed no Git action.

## Scope and safeguards
- Progress report: 2-3 pages; submission Monday 2026-11-16 13:00 Asia/Bangkok. Evidence-complete draft due Sunday 2026-11-15 18:00 Asia/Bangkok.
- Required comparison: fixed class-stratified 1 percent subset (about 120 of 11,959 training images) versus 100 percent training data.
- Use validation only for tuning/selection. Do not inspect test labels or run final test evaluation until decisions are locked and Project Lead authorizes it.
- Experiments run only on Google Colab T4 after committed code is pulled there; never on the team computer.
- Required architecture: modular `.py` files, one clear Python orchestrator, and one thin GitHub-imported Colab notebook limited to setup, repo load/update, config selection, and entry-point invocation.

## Evidence register
| Item | Status | Evidence or owner |
| --- | --- | --- |
| Official provider split totals | Verified handoff | BloodMNIST_224: train 11,959; validation 1,712; test 3,421. Test remains locked. Training Engineer handoff, 2026-11-13 10:28 Asia/Bangkok. |
| Dataset identity | Verified handoff | BloodMNIST_224 MD5: `b718ff6835fcbdb22ba9eacccd7b2601`. Training Engineer handoff, 2026-11-13 10:28 Asia/Bangkok. |
| Fixed 1 percent design | Documented only | `notebooks/01_data_audit_and_subset.md`: seed 2026; stratified largest-remainder allocation. Do not claim saved artifact until verified. |
| Data audit/subset artifact | Awaiting verification | Expected `artifacts/data_audit` and `splits/bloodmnist_train_1pct_seed2026.npz`. |
| Majority baseline and frozen probe | Awaiting | Data and Baseline Engineer. |
| Fine-tuning/scratch validation evidence | Not available | Training Engineer confirmed no committed fine-tuning or scratch-run evidence as of 2026-11-13 10:28 Asia/Bangkok; main at `fded1f25216f00a0cdf09a46be6fe39c8c19dab8` contains no explicitly authorized execution manifest/config. |
| Config/entry point, seed, validation metrics, GPU/runtime, run artifacts | Not available | Cannot truthfully report until committed pipeline is pulled and run in Colab T4. |
| Final test metrics | Not authorized/not run | Must not appear as results. |

## Progress-report outline
1. Problem, data, official split totals, class counts/prevalence, and 1 percent versus 100 percent protocol.
2. Completed methods and compute: only verified model/configuration, seeds, metric, and Colab T4 details.
3. Current evidence table: validation metrics, spread, configuration IDs, runtime, and artifact paths.
4. Completed comparison versus remaining plan, explicitly distinguished.
5. Limitations, remaining work, test safeguard, responsible use, and AI declaration.

## Evidence checklist
- [ ] Train/validation class-count and prevalence CSV/JSON; test total only.
- [ ] 1 percent subset metadata: seed, actual size, class allocations, path, and commit.
- [ ] Per reported run: config, commit, run/date, T4 identity, seed, data size, runtime, and saved metrics.
- [ ] Validation macro-F1 minimum; per-class metrics/confusion matrix only if saved.
- [ ] State whether preliminary single run or 3-seed mean plus sample SD.
- [ ] Figures plus source data and captions with run/config IDs.
- [ ] README module inventory, CLI, configuration, artifact map, split protocol, Colab sequence, and test lock.
- [ ] Decision log for preprocessing, allocation, metric, selection, seeds, compute changes, and deviations.

## Decisions and blockers
| Topic | Status | Action |
| --- | --- | --- |
| Modular Python and thin Colab architecture | Required, not yet verified as implemented | Do not describe as complete until Project Lead confirms module/entry-point/config paths. |
| Training execution evidence | Blocked | No authorized committed execution manifest/config exists at `main@fded1f25216f00a0cdf09a46be6fe39c8c19dab8`; Project Lead must integrate/commit pipeline, then confirm Colab pull/run. |
| Test access | Locked | Exclude test labels and metrics from progress results. |
| Results table | Pending | Populate only from saved engineer artifacts. |

## Handoffs and next action
- Data and Baseline Engineer: provide committed audit/subset and baseline/probe manifests and evidence paths.
- Training Engineer: once pipeline is integrated/committed and Colab pull/run is confirmed, provide validation-only run manifests, metrics, and artifact paths.
- Project Lead: confirm implemented module, entry point, and config paths; review and commit documentation artifacts.
- Next: integrate verified evidence into report tables/captions and README; prepare evidence-complete draft by 2026-11-15 18:00 Asia/Bangkok.
