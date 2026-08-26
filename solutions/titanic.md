# Titanic — the classic first classification (real data) — worked solution

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RGarancs/applied-ai-academy-labs/blob/main/solutions/titanic.ipynb) &nbsp; [← back to the problem set](../titanic.ipynb)

**1,309 rows** · `titanic.csv`

> Every number and chart below is the real output of the code shown.
> Try the problems yourself first — the point is the attempt, not the answer.

The Lecture 2 dataset: predict survival from what was known at boarding.
Key results the notebook derives from the file itself:

* Survival 38% overall; sex × class in %: women 97 / 89 / 49, men 34 / 15 / 15,
  children under 16: 88 / 96 / 41.
* Baselines: everyone-dies 62%, women-live-men-die 78%; logistic regression 80%.
* The 80% model still misses 29% of true survivors — accuracy hides the shape
  of the error, the confusion matrix shows it.
* Odds multipliers: female ×11.6, one class step down ×0.43, each year of age ×0.965.

Canonical source: titanic3, Harrell / Vanderbilt Biostatistics (hbiostat.org/data).
