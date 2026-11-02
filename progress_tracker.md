# Project Progress Tracker

**Project:** Blood Cell Classification with ResNet-18  
**Course:** 2102666 Fundamentals of Deep Learning and Applications  
**Progress-report deadline:** 16 November 2026, 13:00  
**Final presentation and submission:** 7 December 2026

## Status Legend

- [ ] Not started
- [-] In progress
- [x] Complete
- [!] Blocked / needs attention

## Milestones

| Milestone | Due | Status | Notes |
| --- | --- | --- | --- |
| Data audit and reproducible split setup | Before progress report | [ ] | Includes required class-count and prevalence table. |
| Baselines and first deep-model result | Before 16 November | [ ] | Required for the progress report. |
| Progress report | 16 November 2026, 13:00 | [ ] | Two to three pages with first numbers. |
| Full controlled comparison | Before final submission | [ ] | Three strategies at 1% and 100% data sizes, with three seeds. |
| Augmentation comparison | Before final submission | [ ] | Best validation-selected strategy, with and without augmentation. |
| Final test evaluation | After decisions are frozen | [ ] | Test set must be evaluated only once. |
| Presentation, demonstration, final report, and repository | 7 December 2026 | [ ] | Demo must run live; repository reproduces headline result. |

## Data and Reproducibility

| Task | Status | Evidence / output | Notes |
| --- | --- | --- | --- |
| Load official BloodMNIST 224×224 splits | [ ] | Data-loading script or notebook | Official counts: train 11,959; validation 1,712; test 3,421. |
| Verify image shapes, labels, and class names | [ ] | Data-audit output | Confirm eight classes and RGB inputs. |
| Create training class-count and prevalence table | [ ] | Table/CSV/figure | Required in progress report. |
| Create validation class-count and prevalence table | [ ] | Table/CSV/figure | Required in progress report. |
| Record test-set total | [ ] | Counts table | Report total of 3,421 without using it for decisions. |
| Create fixed, stratified 1% training subset | [ ] | Committed indices file and generation script | About 120 images; save random seed and per-class counts. |
| Protect test split from training and tuning | [ ] | Documented evaluation procedure | Do not read the test set until final decisions are frozen. |
| Add data-download instructions | [ ] | README section | Do not commit data unless licence and size permit. |

## Experiment Infrastructure

| Task | Status | Evidence / output | Notes |
| --- | --- | --- | --- |
| Set seeds: 42, 123, and 2026 | [ ] | Configuration and code | Use all three for main-table results. |
| Define shared preprocessing and ImageNet normalization | [ ] | Configuration | Keep consistent across comparable arms. |
| Implement metrics | [ ] | Evaluation script | Per-class precision/recall, macro-F1, macro one-vs-rest ROC-AUC, confusion matrix. |
| Include class prevalence beside reported metrics | [ ] | Results-table template | Required by Track 1. |
| Log loss and macro-F1 training curves | [ ] | Saved plots/logs | Required for first deep-model result in progress report. |
| Save validation-selected checkpoints | [ ] | Checkpointing logic | Select by validation macro-F1 only. |
| Save configs, runtime, and results per run | [ ] | CSV/JSON plus configuration files | Supports reproducibility and compute accounting. |
| Pilot one epoch and estimate compute cost | [ ] | Timing record | Check free Colab quota and under-two-hour run requirement. |

## Required Experiments

### Baselines

| Experiment | 1% subset | 100% training set | Three seeds | Status | Notes |
| --- | --- | --- | --- | --- | --- |
| Majority-class baseline | [ ] | [ ] | Not applicable | [ ] | Report per-class metrics and macro-F1. |
| Frozen ResNet-18 linear probe | [ ] | [ ] | [ ] | [ ] | Cache frozen features for no-augmentation runs. |

### Main controlled comparison — no augmentation

| Strategy | 1% subset | 100% training set | Learning-rate tuning | Three-seed rerun | Status |
| --- | --- | --- | --- | --- | --- |
| Frozen ResNet-18 linear probe | [ ] | [ ] | [ ] | [ ] | [ ] |
| Full ResNet-18 fine-tuning | [ ] | [ ] | [ ] | [ ] | [ ] |
| ResNet-18 from random weights | [ ] | [ ] | [ ] | [ ] | [ ] |

**Control rule:** Within each training-set size, use the same official validation split, preprocessing, batch size, epoch budget, and evaluation procedure for all three strategies. Each strategy receives its own equal-size learning-rate sweep.

### Augmentation comparison

| Task | Status | Notes |
| --- | --- | --- |
| Select best strategy using validation macro-F1 | [ ] | Make the selection before test evaluation. |
| Compare training-only augmentation vs. no augmentation | [ ] | Use horizontal/vertical flips and rotations up to 15°, as proposed. |
| Run selected augmentation comparison with three seeds | [ ] | Report mean ± sample standard deviation. |

## Progress Report Checklist — 16 November

| Required item | Status | Evidence / location |
| --- | --- | --- |
| Training and validation counts table with per-class prevalence | [ ] |  |
| Test total stated as 3,421 | [ ] |  |
| Fixed stratified 1% subset saved in repository | [ ] |  |
| Majority baseline with numbers | [ ] |  |
| Frozen linear-probe baseline with numbers | [ ] |  |
| First deep-model result with numbers | [ ] |  |
| Training curves for the first deep model | [ ] |  |
| Pilot runtime and compute/run-count assessment | [ ] |  |
| Changes from proposal and rationale | [ ] | Explain 1% vs. 100% comparison required by feedback. |
| Remaining-three-weeks plan | [ ] |  |

## Final Submission Checklist

| Required item | Status | Evidence / location |
| --- | --- | --- |
| Main result table includes mean ± sample standard deviation across three seeds | [ ] |  |
| Per-class precision and recall reported | [ ] |  |
| Macro-F1 and macro ROC-AUC reported | [ ] |  |
| Prevalence shown beside every metric table | [ ] |  |
| Test set evaluated only after choices are frozen | [ ] |  |
| Responsible-use section and AI-tool declaration | [ ] |  |
| README contains one-command/notebook reproduction instructions | [ ] |  |
| Environment, configurations, code, and data instructions are included | [ ] |  |
| Headline weights included or download link provided | [ ] | If weights exceed 200 MB, provide a link. |
| Live demo and backup video prepared | [ ] |  |

## Experiment Log

Add one row after each completed run or decision.

| Date | Data size | Strategy | Seed | Best validation macro-F1 | Runtime | Outcome / next action |
| --- | --- | --- | ---: | ---: | --- | --- |
|  |  |  |  |  |  |  |
