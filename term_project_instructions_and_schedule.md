# Term Project

## 2102666

**2102666 Fundamentals of Deep Learning and Applications**  
Department of Electrical Engineering, Chulalongkorn University

### Weeks 7 to 16 — announced Monday 12 October 2026

## Applied deep learning, end to end

**One problem, two baselines, one honest number — and a demonstration that runs.**

Assoc. Prof. Dr. Supavadee Aramvith  
Supavadee.A@chula.ac.th  
Semester 1, Academic Year 2026

This sheet is the complete specification of the term project. The syllabus (§6.1 and §6.2) fixes the weights and the milestones; everything else below is how those are applied this year. References to the lecture notes are written as, for example, Lecture Note 5 §9.

The term project is worth 45 percent of the course. Across the second half of the semester you apply deep learning end to end — from a problem and its data, through baselines and a trained model, to an honest evaluation, a written report and a demonstration that runs in front of the class. The project is done in teams of two. It is the piece of the course that is yours: the labs gave you the parts, the project asks you to build something with them.

## Calendar

All submissions go to myCourseVille. Dates are Mondays, the day of the class.

| Milestone | Date | Weight | What is handed in |
|---|---|---:|---|
| Project announced | 12 October 2026 (Week 7) | — | this sheet; teams form |
| Proposal | 19 October 2026 (Week 8), 13:00 | 5% | one to two pages, template in the section The proposal |
| Feedback on proposals | by 26 October 2026 | — | written comments; a proposal may be sent back once for revision within one week |
| Progress report | 16 November 2026 (Week 12), 13:00 | 5% | two to three pages with your first numbers |
| Presentation and demonstration | 7 December 2026 (Week 15), in class | 15 % | 12 slides at most, a live demonstration, questions to every member |
| Final report and code repository | 7 December 2026, 23:59 | 20 % | six to ten pages, and a repository that reproduces the headline number |

Seven weeks separate the proposal from the presentation. Spend the first four getting data, baselines and a first model to run — the progress report asks for numbers, not plans — and the last three on the comparison, the write-up and the demonstration.

## Teams

Teams are two students. A student who wishes to work alone asks for permission before 19 October and states the reason in the proposal; a one-person project is marked to the same standard, so choose its scope accordingly. Teams of three are not formed this year.

A team shares one mark for each milestone, with two safeguards. The final report ends with a contribution statement — one paragraph per member, in that member’s own words, saying what they did. At the demonstration, every member answers questions on any part of the project. Where the statement and the demonstration show that contributions were unequal, individual marks are adjusted.

## Choosing a problem

There are two routes. Both lead to the same requirements, the same template and the same rubric.

**Route A — a track.** Choose one of the six tracks in the next section. Each names a dataset that is public and downloadable in Colab, the baselines you must beat, the one comparison you must run, and the metric. The framing is done for you so that the week between announcement and proposal goes into reading the data, not searching for it. You may change the dataset within a track if you have access to a better one and can show it in the proposal.

**Route B — your own problem.** Propose a problem of your own, including one from your thesis or your work, on the same proposal template. It is approved when it meets the same requirements as a track: data you already have access to on 19 October, a trivial and a simple baseline, a metric fixed before training, one controlled comparison, and a run that fits the compute budget. Two conditions for thesis data: the project must be a self-contained piece of work with its own baseline and comparison, and the proposal must state what already existed before 12 October — code, trained models, results — so that the project is marked on what is new.

Teams are encouraged to discuss their choice before 19 October, after class on 12 October or by appointment.

## The tracks

Tracks 1 to 4 use material taught by Week 9. Tracks 5 and 6 depend on lectures given after the proposal (Weeks 10–11 and 12–13); choose them only if you are willing to read ahead in Prince and d2l.ai, since your progress report falls on the week the material is taught.

### Track 1 — Image classification by transfer learning

- **Problem.** Classify medical or natural images with a pretrained backbone, when the training set is small.
- **Data.** One of the MedMNIST v2 collections at 224 × 224 (pip install medmnist; for example PneumoniaMNIST, DermaMNIST, OrganAMNIST, or BloodMNIST), or a comparable public set you name.
- **Baselines to beat.** Majority class; a linear probe on frozen ResNet-18 features (Laboratory 7, Part 2).
- **Required comparison.** Linear probe vs full fine-tuning vs training from random weights, same data, same budget — then the best of these with and without augmentation (Lecture Note 7 §2).
- **Metric.** Per-class precision and recall, macro-F1 and ROC-AUC, with the class prevalence stated beside every number (Lecture Note 5 §3.1 and §4.3). Accuracy alone is not a result on an imbalanced set.
- **Stretch.** Class-imbalance remedies (re-weighting, re-sampling); a saliency map on the images the model gets wrong.

### Track 2 — Detection or segmentation on a small dataset

- **Problem.** Locate or delineate objects, not just name them.
- **Data.** Oxford-IIIT Pet (segmentation masks, in torchvision.datasets), the Penn-Fudan pedestrian set (170 images with boxes and masks), or a small set of your own with at least 150 annotated images.
- **Baselines to beat.** For segmentation: predict the training-set majority label at every pixel; a U-Net trained from random weights. For detection: a pretrained detector used as is, with no fine-tuning.
- **Required comparison.** Pretrained vs randomly initialised encoder, same training budget; then your best model with and without augmentation.
- **Metric.** Mean IoU per class (segmentation) or mAP at IoU 0.5 (detection), as defined in Lecture Note 7 §6 and §9.2.
- **Stretch.** Dilated convolutions (Lecture Note 7 §10.3) or a different decoder; a failure analysis by object size.

### Track 3 — Time-series forecasting

- **Problem.** Forecast a measured quantity several steps ahead.
- **Data.** The ETT electricity-transformer series (ETTh1, GitHub zhouhaoyi/ETDataset), the UCI Beijing PM2.5 set, or a Thai series — load, traffic, air quality, prices — whose access you have confirmed by 19 October.
- **Baselines to beat.** Persistence (repeat the last value) and seasonal naïve (repeat the value one period ago); then a linear model on the same lookback window. No recurrent network is reported until these three have numbers.
- **Required comparison.** LSTM or GRU vs the linear baseline at three horizons; then the lookback window varied (short, medium, long) with everything else fixed.
- **Metric.** MAE and RMSE at each horizon, on a rolling-origin evaluation. Scalers are fitted on the training segment only (Lecture Note 5 §8.2 and §8.5).
- **Stretch.** A small Transformer encoder (Week 9); exogenous features in and out.

### Track 4 — Text classification with a pretrained Transformer

- **Problem.** Assign a label to a short text — sentiment, topic, intent.
- **Data.** Thai: Wisesight Sentiment (GitHub PyThaiNLP/wisesight-sentiment, about 26,000 messages) or Wongnai reviews. English: AG News or IMDB. Or a labelled set of your own with at least 2,000 examples.
- **Baselines to beat.** Majority class; TF-IDF features with logistic regression (scikit-learn).
- **Required comparison.** Frozen encoder plus a linear head vs full fine-tuning of a pretrained model (WangchanBERTa for Thai, DistilBERT for English); then a learning curve — the fine-tuned model trained on 10 %, 30 % and 100 % of the training set.
- **Metric.** Macro-F1 and the confusion matrix, with class prevalence stated.
- **Stretch.** Error analysis by text length; a comparison of two tokenisers on Thai text.

### Track 5 — A question-answering application on a document collection (Weeks 10–11 material)

- **Problem.** Build an assistant that answers questions from a fixed set of documents — a manual, a set of regulations, the lecture notes of this course — using a language model with retrieval.
- **Data.** The document collection, and an evaluation set of at least 50 questions with reference answers written by you before the system is built.
- **Baselines to beat.** The language model alone, no retrieval; retrieval alone — the single best passage returned, unedited.
- **Required comparison.** Retrieval-augmented vs the two baselines on your evaluation set; then one retrieval setting varied (passage length, number of passages, or embedding model) with everything else fixed.
- **Metric.** Answer correctness marked against the reference by a written rubric, by two team members independently, with agreement reported; retrieval recall at k. Cost and latency per question are reported.
- **Models.** An open-weight model that runs in Colab, or an API with the spending logged and reported.

### Track 6 — A small generative model (Weeks 12–13 material)

- **Problem.** Learn to generate samples from an image distribution and show, with numbers, how well.
- **Data.** Fashion-MNIST, a 64 × 64 subset of CelebA, or a small image set of your own with at least 5,000 images.
- **Baselines to beat.** A plain autoencoder with the same latent size (for reconstruction); samples drawn from the training set itself (the ceiling on any sample-quality score).
- **Required comparison.** VAE vs GAN — or vs a small diffusion model — at equal parameter count and training budget; then one design choice varied (latent size, or the number of diffusion steps).
- **Metric.** FID on at least 5,000 samples, and reconstruction error where it applies; a grid of samples with the nearest training images shown beside them.
- **Warning.** This track is taught two weeks before the presentation. Choose it only if you can start from Prince chapters 16–18 and the d2l.ai chapters on your own in October.

## What every project must have

These are the requirements the rubric is scored against. Lecture Note 5 §9 — the ten checks — is the list the reports will be read with; apply it to your own work before you hand it in.

1. **A problem, stated.** What is predicted, from what, for whom, and what a wrong prediction costs. One paragraph; a reader who knows nothing of your field should understand it.
2. **Data you can describe.** Source, licence, size, how it was collected, and the training / validation / test split, made once, before any model is trained. Class prevalence or the target’s distribution, in a table.
3. **Two baselines before any network.** A trivial baseline (majority class, persistence, the training mean) and a simple one (logistic regression, a linear model, a frozen probe). A deep model that does not beat them is a result worth reporting — honestly (Lecture Note 5 §7.2).
4. **A model whose design is justified** from the lectures: why this architecture for this data, why this loss, why this optimiser and schedule.
5. **One controlled comparison.** One thing changed at a time; every arm gets the same training budget and its own learning-rate sweep (Lecture Note 5 §7); otherwise the comparison is not reported.
6. **Seeds and spread.** At least three seeds for every number in the main table; mean and spread reported, never the best. A difference smaller than the spread is not a difference.
7. **The test set read once,** after every choice has been made on the validation set, and said so in the report.
8. **A compute budget a classmate can afford.** Every reported run must be reproducible on a free Colab T4 in under two hours. The research unit’s GPU servers can be arranged for larger runs; ask in the proposal.
9. **Reproducibility.** A repository with a README, the environment, fixed seeds, configuration files, instructions for obtaining the data, and one command (or one notebook) that reproduces the headline number.
10. **Responsible use.** One section of the report: where the data came from and whether you were allowed to use it; who is affected if the model is wrong; what the model must not be used for; and a declaration of every AI tool used and for what (syllabus §11). The Week 14 ethics discussion will take its case studies from your projects.

## The proposal (19 October, 5 %)

One to two pages, PDF, under these headings in this order:

1. Title, team members, route (A — which track, or B).
2. Problem — three to five sentences, as in requirement 1.
3. Data — source, licence, size, access confirmed (attach a screenshot or a sample row / image), and your intended split with counts.
4. Baselines — the trivial and the simple one, named.
5. Model — what you intend to train, in one paragraph, with the lecture it comes from.
6. Evaluation — the metric, fixed now; the number of seeds; how the test set will be protected.
7. The comparison — the one thing you will vary and what you expect to see.
8. Plan — one line per week from 19 October to 7 December.
9. Risks and fallback — what could stop you, and what you would do instead.
10. Compute — Colab, the laboratory servers, or your own; estimated hours per run.
11. Route B only: what already exists — code, models, results from before 12 October.

Marked on: a problem that is clear and bounded; data you actually have; an evaluation plan written before training; a plan that fits seven weeks. A proposal that is sent back is revised within one week with no change to its mark.

## The progress report (16 November, 5 %)

Two to three pages. Data loaded and split, with the counts table; both baselines with numbers; the first deep-model result and its training curves; what has changed since the proposal and why; the plan for the remaining three weeks. A project with no number by Week 12 is in trouble — say so, and we re-scope it together that week. Unflagged trouble discovered on 7 December is marked as it is.

## The presentation and demonstration (7 December, 15 %)

**Format.** Five teams, 25 minutes each: 15 minutes of presentation, 5 minutes of live demonstration, 5 minutes of questions. At most 12 slides. Both members present, and both answer questions.

The presentation follows the report: problem and why it matters (2 slides), data and split (1–2), baselines and model (2–3), the main table with seeds and spread (1), the comparison (1–2), what went wrong and what you learned (1), responsible use (1).

The demonstration runs the trained model live on an input the audience chooses — an image, a sentence, a date range, a question. Bring a two-minute recorded video as insurance against the network; a demonstration that is video only loses half the demonstration mark. A result that cannot be demonstrated is not demonstrated.

## The final report and repository (7 December, 23:59, 20 %)

**The report.** Six to ten pages excluding references and appendix, PDF, in this structure: abstract (150 words); 1 Problem; 2 Data; 3 Method; 4 Experiments — setup, baselines, main result, the comparison, seeds and spread, the single test-set reading; 5 Discussion and limitations; 6 Responsible use and AI-tool declaration; 7 Contribution statement; references; appendix with every configuration. Every number in the text appears in a table or a figure. Every figure has axes labelled with units.

**The repository.** GitHub (private, with the instructor added) or a zip on myCourseVille: README with the reproduce command, requirements.txt or an environment file, configuration files, the training and evaluation code, the executed notebook that produced the figures, and instructions for obtaining the data (the data itself only if it is small and its licence allows). Trained weights for the headline model if under 200 MB; otherwise a download link.

## How the project is marked

The syllabus criteria (§6.2) are applied to the presentation, the report and the repository together — 35 of the 45 percent. The proposal and the progress report carry their own short criteria, given above.

| Criterion | Weight | What earns it |
|---|---:|---|
| Problem formulation and positioning | 20 % | a bounded problem, a reader who understands why it matters, and the related approaches placed in two paragraphs |
| Technical soundness of the method | 25 % | design choices justified from the lectures; the loss, the optimiser, the schedule and the regularisation each chosen for a reason |
| Experimental rigour | 30 % | both baselines; one fair, controlled comparison; three seeds with spread; the test set read once; prevalence beside every metric |
| Reproducibility | 10 % | the headline number reproduced from the README in one command; configurations complete |
| Communication | 15 % | a report that reads in order, figures that stand alone, a presentation that fits its time, a demonstration that runs |

## Rules

- **Late work.** 10 percent of the milestone mark per calendar day, for at most three days, after which it is not accepted (syllabus §11). The presentation cannot be late: a team that does not present on 7 December receives no presentation mark except for documented medical reasons.
- **Integrity.** Every reported number comes from an experiment you ran. Code you did not write — libraries, tutorials, a pretrained model — is cited in the README and the report. Fabricated or copied results are academic misconduct.
- **AI tools** may be used for learning, coding, debugging and language editing, and every use is declared in section 6 of the report. You remain able to explain every line you submit; the questions at the demonstration will test this.
- **Data.** Use data whose licence permits it. No personal data without consent and anonymisation. The test set is not touched until the final reading.
- **Scope changes** after 19 October are allowed once, in writing, before 16 November.

## Help

Office hours by appointment through myCourseVille or the course LINE group. Questions whose answer helps everyone are answered in the LINE group. The laboratory’s GPU servers are available to teams whose proposals ask for them.
