# Project State

**Project:** Blood Cell Classification with ResNet-18
**Course:** 2102666 Fundamentals of Deep Learning and Applications
**Last updated:** 13 November 2026, 10:26 Asia/Bangkok
**Git authority:** Project Lead exclusively manages repository access, status, branches, pulls, commits, pushes, and Colab commit pinning. Teammates must not perform Git operations unless explicitly delegated a named action.

## Current status

**Phase:** Progress-report implementation and evidence collection.
**Overall:** Data-audit workflow is committed. Controlled-comparison protocol and analyst state are approved for integration in the pending documentation commit; required generated audit evidence, fixed subset artifact, baselines, shared training pipeline, Colab launcher, and progress-report results remain outstanding.

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
| Current verified commit | `83e002a359e9ac9a15855683ec59c36c6874dc40` — `data audit` |
| Working tree at verification | `PROJECT_STATE.md`, controlled-experiment protocol, and analyst state pending Lead commit |
| Intended Colab baseline | **No experiment commit is designated.** Do not run Colab experiments from `83e002a`; wait for a Lead-announced implementation commit. |

## Decisions and rationale

| Decision | Rationale | Status |
| --- | --- | --- |
| Dataset: BloodMNIST, official provider splits | Approved proposal; comparable, fixed public splits. | Confirmed |
| Data sizes: 1% and 100% training | Required by proposal feedback; makes limited-data comparison meaningful. | Confirmed |
| 1% subset: fixed, class-stratified, seed 2026 | Ensures fair, reproducible comparisons across model seeds and strategies. | Confirmed in committed notebook/docs; artifact not yet generated/committed. |
| Required strategies | Majority baseline; frozen ResNet-18 linear probe; full ResNet-18 fine-tuning; ResNet-18 from random weights. | Confirmed |
| Model selection | Validation macro-F1 only; final test evaluation only after decisions freeze. | Confirmed |
| Compute environment | Google Colab T4 only; no team-computer experiment execution. | Confirmed |
| Code/notebook design | Modular `.py` files and one thin GitHub-imported Colab launcher limited to setup, repository update, config selection, and entry-point invocation. | Confirmed; not yet implemented. |

## Evidence and artifacts

| Artifact | Location | State |
| --- | --- | --- |
| Proposal | `Proposal_Blood Cell Classification with ResNet.md` / `.pdf` | Committed |
| Proposal feedback | `Proposal_Feedback.md` | Committed |
| Course instructions | `term_project_instructions_and_schedule.md` | Committed |
| Progress checklist | `progress_tracker.md` | Committed; not yet updated for completed evidence |
| Data audit notebook | `notebooks/01_data_audit_and_subset.ipynb` | Committed at `83e002a`; currently exceeds the new thin-notebook standard and must be refactored/replaced by a thin launcher plus modular code. |
| Data audit documentation | `notebooks/01_data_audit_and_subset.md` | Committed |
| Report-ready official split counts | `artifacts/data_audit/official_split_counts.md` | Not present/committed |
| Fixed 1% subset indices | `splits/bloodmnist_train_1pct_seed2026.npz` | Not present/committed |
| Training/evaluation code and configs | — | Not present/committed |
| Colab launcher | — | Not present/committed |
| Controlled experiment protocol | `docs/experiments/controlled_experiment_protocol.md` | Reviewed and approved; pending Lead commit |
| Experiment analysis state | `docs/state/experiment_analysis_status.md` | Reviewed and approved; pending Lead commit |

## Verified dataset facts

- Official BloodMNIST splits: train **11,959**, validation **1,712**, test **3,421**.
- Eight classes; RGB source inputs.
- Test labels must remain unused until final decisions are frozen.

## Work ownership, handoffs, and dependencies

| Owner | Assignment | Current dependency / handoff |
| --- | --- | --- |
| Project Lead | Git workflow, integration, state, timeline, commit pinning | Repository verified; teammate work must be delivered without teammate Git actions. |
| Data & Baseline Engineer | Dataset audit, fixed splits, majority and frozen-linear-probe baselines | Generate audit evidence and fixed subset; stage/supply artifacts for lead integration. |
| Training Engineer | Shared pipeline and Colab T4 fine-tuning | Needs committed modular data/split interfaces and lead-issued commit reference. |
| Report & Reproducibility Lead | Progress report, evidence package, reproducibility docs | Needs counts, baseline/deep-model results, runtime and committed workflow. |
| Experiment Coordinator & Analyst | Controlled matrix, seed aggregation, augmentation ablation, validation decisions | Protocol approved for the pending documentation commit; awaits shared pipeline/configs and fixed-split interface. |

## Blockers and next actions

| Priority | Blocker | Owner | Next action |
| --- | --- | --- | --- |
| Critical | Progress-report evidence is not yet generated: counts artifact, fixed subset, majority baseline, linear probe, first deep result, curves, runtime estimate. | Data/Training/Report leads coordinated by Project Lead | Generate only via Colab T4, deliver artifacts for lead review and commit. |
| Critical | Current committed data-audit notebook is not compliant with the thin-notebook/modular-code constraint. | Project Lead with Data/Training | Refactor workflow into documented modules plus a thin GitHub-imported Colab launcher before experiment runs. |
| High | No current committed training/evaluation modules, configs, result schema, or launcher. | Training Engineer / Project Lead | Supply modular implementation and configs; lead integrates and pins commit. |
| High | Controlled protocol needs exact data-split and pipeline interface alignment before execution. | Data & Baseline Engineer / Training Engineer / Experiment Coordinator | Provide split paths/checksum contract and CLI/config/result schema; Lead reviews any protocol-impacting change. |
| High | No dated run plan is yet recorded for the 16 November 2026, 13:00 deadline. | Project Lead / Report Lead | Create evidence/run schedule immediately after implementation handoff. |

## Operating procedure

1. Teammates stage or provide artifacts and state updates without Git commands.
2. Project Lead reviews, integrates, commits, pushes, and announces the exact commit.
3. Colab clones or updates to that exact announced commit, then invokes the thin launcher with a named configuration.
4. Every experiment records configuration, seed, runtime, validation metrics, artifact locations, and commit reference.
5. Update this file and relevant documentation whenever status, decisions, evidence, dependencies, or blockers change.
