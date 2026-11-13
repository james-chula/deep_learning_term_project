# Progress Report Evidence Template

**Use:** Populate only with verified, saved validation evidence. Replace every bracketed field; delete unverified rows. Do not report test metrics.

## Table 1. Data and split protocol
| Split/condition | Images | Per-class prevalence source | Selection role | Evidence path |
| --- | ---: | --- | --- | --- |
| Official training (100 percent) | 11,959 | [audit CSV/JSON] | model fitting | [path] |
| Fixed training subset (1 percent) | [actual size] | [subset metadata] | model fitting | [path] |
| Official validation | 1,712 | [audit CSV/JSON] | tuning/checkpoint/strategy selection | [path] |
| Official test | 3,421 | total only | protected; not used for this report | [source] |

## Table 2. Verified validation evidence
| Condition | Method | Config ID | Seed(s) | Validation macro-F1 | Other saved metrics | Runtime | Colab GPU | Run/artifact path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [1 percent/100 percent] | [majority/probe/fine-tune/scratch] | [ID] | [seed or n=3] | [value or mean +/- sample SD] | [only saved metrics] | [min] | [T4 identity] | [path] |

## Figure placeholders
- Figure 1: training/validation class prevalence. Source: [CSV/JSON]. Include y-axis in percent and caption with split totals.
- Figure 2: validation macro-F1 by verified method and data condition. Error bars only for 3+ seed runs; label preliminary single runs explicitly.
- Figure 3 (optional): validation learning curves or confusion matrix from one identified run. Include axes/units and artifact path.

## Narrative guardrails
- Say planned, scheduled, or pending for work without a completed saved artifact.
- Distinguish a single preliminary seed from main-comparison mean plus sample standard deviation.
- State the exact validation-selection rule and that final test evaluation is deferred.
- Cite source commit, config, run metadata, and artifact path for every number/figure.

## Required provenance fields per reported run
1. source commit; 2. config path and resolved settings; 3. run ID/date; 4. Colab GPU/runtime identity; 5. seed; 6. train condition and exact image count; 7. validation split; 8. command/entry point; 9. metrics file; 10. figure/checkpoint paths.
