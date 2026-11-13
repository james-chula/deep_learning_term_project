# Project State

**Project:** Blood Cell Classification with ResNet-18
**Course:** 2102666 Fundamentals of Deep Learning and Applications
**Last updated:** 13 November 2026, 11:34 Asia/Bangkok
**Git authority:** Project Lead exclusively manages repository access, status, branches, pulls, commits, pushes, and Colab commit pinning. Teammates must not perform Git operations unless explicitly delegated a named action.

## Current status

**Phase:** Progress-report implementation and evidence collection.
**Overall:** Simplified procedural validation-only fine-tuning is committed and pushed. The designated launcher checks out its reviewed code and documentation commit before invoking the simple root entry point. No Colab experiment has run.

## Timeline

| Milestone | Due (Asia/Bangkok) | Status | Next requirement |
| --- | --- | --- | --- |
| Data audit and reproducible split setup | Before progress report | In progress | Generate and commit report-ready counts and fixed stratified 1% indices. |
| Baselines and first deep-model result | Before 16 November 2026, 13:00 | Not started | Run only in Google Colab with T4 GPU; collect results and curves. |
| Progress report | 16 November 2026, 13:00 | At risk | Counts, fixed 1% subset, two baselines, first deep-model evidence, runtime assessment. |
| Controlled comparison and augmentation study | Before final submission | Not started | Validation-only selection; three seeds for headline comparisons. |
| Presentation/demo/final report/repository | 7 December 2026, 23:59 (submission); presentation in class | Not started | Reproducible headline run and live demo. |

## Repository and commit reference

| Item | Current state |
| --- | --- |
| Local repository | `/home/rakazo/deep_learning_term_project` |
| Remote | `git@github.com:james-chula/deep_learning_term_project.git` |
| Branch | `main` tracking `origin/main` |
| Verified current `main` and `origin/main` | `86ec8694f578e0e99e291f4a786f0e8d3c105c6c` |
| Baseline implementation | `5db586c7b37b3585051c88379502ab6c9f2e546c` |
| Stabilized provider/API commit | `8a637d34017bce6311225b82fd5d3a6563d7dd26` — canonical provider interface added after static review; no data load or experiment occurred. |
| Baseline launcher commit | `74ca0681aa21fdedcbd2a85f9793f71c2ad4174f` — thin launcher checks out approved baseline code SHA `8a637d34017bce6311225b82fd5d3a6563d7dd26`; no result evidence exists. |

## Decisions and rationale

| Decision | Rationale | Status |
| --- | --- | --- |
| Dataset: BloodMNIST, official provider splits | Approved proposal; comparable, fixed public splits. | Confirmed |
| Data sizes: 1% and 100% training | Required by proposal feedback; makes limited-data comparison meaningful. | Confirmed |
| 1% subset: fixed, class-stratified, seed 2026 | Ensures fair, reproducible comparisons across model seeds and strategies. | 120 indices; SHA-256 `11d34a47d4a048a954b3ade270e5abda003f181237b24d34e07fe974ef9c035d`. |
| Required strategies | Majority baseline; frozen ResNet-18 linear probe; full ResNet-18 fine-tuning; ResNet-18 from random weights. | Confirmed |
| Model selection | Validation macro-F1 only; final test evaluation only after decisions freeze. | Confirmed |
| Compute environment | Google Colab T4 only; no team-computer experiment execution. | Confirmed |
| Code/notebook design | Modular `.py` files and one thin GitHub-imported Colab launcher limited to setup, repository update, config selection, and entry-point invocation. | Confirmed |
| Canonical provider API | `bloodmnist_baselines.build_provider_datasets(*, audit_manifest_path, image_size, train_transform, val_transform, subset_manifest_path=None) -> ProviderDatasets(train, val, metadata)` | Fixed. Do not add another loader or change to `build_train_val`. |

## Evidence and artifacts

| Artifact | Location | State |
| --- | --- | --- |
| Proposal | `Proposal_Blood Cell Classification with ResNet.md` / `.pdf` | Committed |
| Proposal feedback | `Proposal_Feedback.md` | Committed |
| Course instructions | `term_project_instructions_and_schedule.md` | Committed |
| Progress checklist | `progress_tracker.md` | Committed; not yet updated for completed evidence |
| Data audit notebook | `notebooks/01_data_audit_and_subset.ipynb` | Committed; audit evidence only, not the execution launcher. |
| Data audit documentation | `notebooks/01_data_audit_and_subset.md` | Committed |
| Report-ready audit artifacts | `artifacts/data_audit/audit_manifest.json`, `train_class_counts.csv` | Committed; source is official 28px archive resized deterministically to 224px. |
| Fixed 1% subset indices | `splits/bloodmnist_train_1pct_seed2026.npz` | Committed; 120 entries, seed 2026. |
| Mandatory baseline code/config | `src/bloodmnist_baselines/`, `configs/baseline_validation.yaml` | Committed; no results yet. |
| Baseline Colab launcher | `notebooks/colab_run_baselines.ipynb` | Committed thin launcher; no execution evidence. |
| Controlled experiment protocol | `docs/experiments/controlled_experiment_protocol.md` | Reviewed and approved; pending Lead commit |
| Experiment analysis state | `docs/state/experiment_analysis_status.md` | Reviewed and approved; pending Lead commit |

## Verified dataset facts

- Official BloodMNIST splits: train **11,959**, validation **1,712**, test **3,421**.
- Eight classes; RGB source inputs.
- Test labels must remain unused until final decisions are frozen.
- Execution source is official 28px BloodMNIST; resizing to 224px is deterministic. The 224px archive checksum is reference-only.

## Work ownership, handoffs, and dependencies

| Owner | Assignment | Current dependency / handoff |
| --- | --- | --- |
| Project Lead | Git workflow, integration, state, timeline, commit pinning | Repository verified; teammate work must be delivered without teammate Git actions. |
| Data & Baseline Engineer | Dataset audit, fixed splits, majority and frozen-linear-probe baselines | Provider contract is fixed; await explicit review request. |
| Training Engineer | Shared pipeline and Colab T4 fine-tuning | Must correct provider-module nomination in staged source, then submit for Lead re-review. |
| Report & Reproducibility Lead | Progress report, evidence package, reproducibility docs | State reconciled; awaits execution evidence. |
| Experiment Coordinator & Analyst | Controlled matrix, seed aggregation, augmentation ablation, validation decisions | Awaits shared pipeline/configs and fixed-split interface. |

## Blockers and next actions

| Priority | Blocker | Owner | Next action |
| --- | --- | --- | --- |
| Critical | Progress-report experiment evidence is not generated: baseline metrics, first deep result, curves, and runtime estimate. | Project Lead / team | Generate only after static gates, in Colab T4, then commit artifacts. |
| High | Fine-tuning pipeline is staged only. Its CLI and notebook pass the nonexistent provider module `bloodmnist_baselines.provider`. | Training Engineer | Change module argument to `bloodmnist_baselines`; no wrapper module, Git, or experiment. |
| High | Fine-tuning implementation needs full Lead static review and integration. | Project Lead | Review corrected staged source, integrate, commit/push, and pin notebook to resulting SHA. |
| High | No dated run plan is recorded for the 16 November 2026, 13:00 deadline. | Project Lead / Report Lead | Create evidence/run schedule after training handoff. |

## Operating procedure

1. Teammates stage or provide artifacts and state updates without Git commands.
2. Project Lead reviews, integrates, commits, pushes, and announces the exact commit.
3. Colab clones or updates to that exact announced commit, then invokes the thin launcher with a named configuration.
4. Every experiment records configuration, seed, runtime, validation metrics, artifact locations, and commit reference.
5. Update this file and relevant documentation whenever status, decisions, evidence, dependencies, or blockers change.

## Lead review update — 13 November 2026, 10:38 Asia/Bangkok

- The original staged training bundle was not executable: it required a missing provider interface and deliberately raised `NotImplementedError`. It also lacked the completed training/validation engine, metrics, transforms, and reporting integration.
- The original launcher was a `.py` file, not a compliant thin GitHub-imported Colab notebook. No training commit or Colab run was authorized.

## Stabilization checkpoint — 13 November 2026, 10:46 Asia/Bangkok

**Purpose:** Establish one verified repository baseline before any further training integration or Colab execution.

| Item | Verified state | Decision / next action |
| --- | --- | --- |
| Remote synchronization | `main` and `origin/main` are now verified at `86ec8694f578e0e99e291f4a786f0e8d3c105c6c`. | Lead only may commit/push. |
| Baseline package | Present from baseline implementation commit; canonical provider interface is integrated. | Retain as the sole baseline/data implementation. No baseline experiment has run. |
| Fixed subset | Tracked manifest SHA-256 verified as `11d34a47d4a048a954b3ade270e5abda003f181237b24d34e07fe974ef9c035d`. | Immutable authoritative 1% membership definition. |
| Canonical data source | Audit manifest identifies official 28px archive MD5 `7053d0359d879ad8a5505303e11de1dc`; ResNet code must resize deterministically to 224px. | Do not introduce another loader or split. |
| Canonical provider interface | `bloodmnist_baselines.build_provider_datasets(...) -> ProviderDatasets`. It returns no test dataset and records `test_policy=locked_not_loaded`. | Sole approved provider API. |
| Colab execution | No Colab URL, GPU evidence, result JSON, or metrics exist. | Baseline execution is eligible only through the pinned baseline launcher; no execution is underway. |

## Provider-path review — 13 November 2026, 10:56 Asia/Bangkok

**Finding:** the staged Training CLI/notebook provider path is invalid.

- Integrated `src/bloodmnist_baselines/` contains `__init__.py`, `data.py`, `baselines.py`, `metrics.py`, and `runner.py`; it contains **no** `provider.py`.
- `bloodmnist_baselines.__init__` exports `build_provider_datasets` directly.
- The staged training adapter correctly resolves a named module's `build_provider_datasets` attribute, but its CLI/notebook nominate `bloodmnist_baselines.provider`, which will fail with `ModuleNotFoundError`.
- Required correction: use `--provider-module bloodmnist_baselines`. Do not add a duplicate wrapper module merely to mask this mismatch.

### Stable-base gate

Before any experiment authorization: (1) re-verify `main == origin/main` and a clean tree; (2) require every thin notebook to contain the exact approved full SHA; (3) complete a static review of any training integration against the fixed provider API; (4) commit and push the reviewed training integration; (5) update this file with the new expected/current commit. Test data remain locked.

## Commit and launcher designation — 13 November 2026, 11:09 Asia/Bangkok

- Training code commit: `693f658c6c48318f88377e6552e9117cf4aa7a02`.
- Thin Colab launcher commit: `7813d4b50132a1178864a4f1a48df82d0927c233`. Its checkout cell pins the training code commit above, not an unreviewed branch tip.
- Static verification passed for training Python sources, notebook JSON, configuration loading, and direct provider-module resolution up to the unavailable optional MedMNIST dependency. No runtime data/model operation was performed.
- Current next action: Project Lead pushes the commits; a future authorized run must use the Colab T4 launcher and retain generated validation-only artifacts.

## Simplified procedural training implementation — 13 November 2026, 11:34 Asia/Bangkok

**Decision:** User approved replacing the hard-to-read training execution path with a small procedural workflow. The prior modular package remains in the repository as historical integration work, but the designated first-run entry point is now the following root-level files:

- `config.py`: one editable plain dictionary for experiment settings;
- `data.py`: direct import of `bloodmnist_baselines.build_provider_datasets`, creation of training and validation loaders, and verification that `test_policy=locked_not_loaded`;
- `model.py`: pretrained ResNet-18 with an 8-class output layer;
- `train.py`: direct `train_one_epoch` and `validate` functions;
- `run_experiment.py`: simple top-to-bottom T4-only execution, validation macro-F1 selection, and basic artifacts.

**Rationale:** The project team is out of practice with Python. The new entry point eliminates the CLI, YAML configuration, dynamic provider import, orchestration layers, resume/fingerprint machinery, and event abstraction from the learning path while retaining required split safety, validation-only selection, T4 enforcement, Git provenance, metrics, curve, and checkpoints.

**Safety invariants retained:** official 28px BloodMNIST data are resized to 224px; the provider owns audit/split/subset membership; optional committed subset is training-only; validation is official and complete; no test object is requested, constructed, or returned; best model selection uses validation macro-F1 only; no experiment is run outside Colab T4.

**Expected/current implementation commit before launcher/document commit:** `9dc2b4b7d8f34e35bdd80d1afd44077e0becb6a0` (`feat: add simple procedural training workflow`).

**Evidence:** static Python compilation for all five simple files passed. No data download, model download, GPU runtime, metrics, checkpoint, or experiment was created.

**Next action:** finish static review of simplified launcher/documentation, commit/push them, then await explicit Colab execution authorization. Last updated: 13 November 2026, 11:34 Asia/Bangkok.

## Simplification integration complete — 13 November 2026, 11:34 Asia/Bangkok

- Procedural training implementation commit: `9dc2b4b7d8f34e35bdd80d1afd44077e0becb6a0`.
- Documentation, pinned Colab dependencies, provenance capture, and static checks commit: `c1ed80ade605c1613dd2ce1453be67bb8b2644ef`.
- Thin launcher commit: `e1d5fbc59620f6687de7b5440c776db381d64dc0`; it checks out the reviewed code-and-documentation commit `c1ed80ade605c1613dd2ce1453be67bb8b2644ef` and invokes `run_experiment.py`.
- `main` and `origin/main` were pushed and verified at `e1d5fbc59620f6687de7b5440c776db381d64dc0` before this state-record update.
- Static verification: Python compilation passed; `python3 -m unittest discover -s tests -v` passed 4 tests; notebook JSON/pin checks passed; no experiment was run.
- Open execution blocker: explicit user authorization for a Google Colab NVIDIA T4 run. Do not run locally. Last updated: 13 November 2026, 11:34 Asia/Bangkok.
