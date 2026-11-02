# Proposal Feedback

⭐⭐⭐⭐☆

## Decision

**Approved.** Items 1–2 below must appear in the progress report.

---

## Required

1. **The counts table:** images per class in the training and validation splits, with prevalence, and the test total (official split: 11,959 / 1,712 / 3,421).

2. **Make the limited-data question real, or drop it.** With about 12,000 training images, a network trained from random weights also does well, and the three arms may differ by less than the seed spread. Run the comparison at two training-set sizes — for example 1% (about 120 images) and 100% — with a fixed, class-stratified subset saved in the repository. Small subsets train in minutes, so the cost is low.

---

## Points to watch

- **The linear probe can be cheap:** compute the frozen features once and train the head on the cached features — seconds per run instead of your estimated 0.25–0.75 GPU hours (without augmentation only).

- **Count the runs** — about 24 by your design — against your free Colab quota once the pilot epoch is timed.

---

*Last graded on 31-10*
