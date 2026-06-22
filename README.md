# Resume–JD Matcher with Gap Analysis

A deep learning system that takes a candidate resume and a job description,
predicts a fit score (No Fit / Potential Fit / Good Fit), and explains the
prediction by surfacing skills the JD asks for that the resume doesn't show.

## Project status
- [ ] Week 1 — Data exploration & preprocessing pipeline
- [ ] Week 2 — Baseline (TF-IDF + classic ML)
- [ ] Week 3 — Fine-tuned transformer classifier
- [ ] Week 4 — Skill extraction + gap analysis module
- [ ] Week 5 — Pipeline integration + Gradio UI
- [ ] Week 6 — Deployment + write-up

## Folder structure

```
resume-jd-matcher/
├── data/
│   ├── raw/            # original downloaded data, untouched (gitignored)
│   ├── processed/      # cleaned/preprocessed data ready for modeling
│   └── external/       # supplementary datasets (extra resumes, skill lists, etc.)
├── notebooks/           # exploration & experimentation (EDA, prototyping)
├── src/
│   ├── data/            # loading & preprocessing scripts (the "real" pipeline,
│   │                     #   once you've prototyped it in notebooks/)
│   ├── models/          # baseline, classifier, and gap-analysis model code
│   ├── utils/            # shared helpers (metrics, plotting, etc.)
│   └── app/              # the Gradio app for the final demo
├── models/               # saved model checkpoints (gitignored — too large for git)
├── reports/
│   └── figures/           # EDA plots, evaluation charts, etc.
└── tests/                  # basic sanity tests for src/ code
```

## Why notebooks/ AND src/?
Notebooks are for exploring and prototyping — messy, iterative, full of dead ends.
Once a piece of logic works and you'll reuse it (e.g., the preprocessing
function, the gap-analysis comparison), move a cleaned-up version into `src/`
as a proper function/module. This is what makes Week 5 (integration) much less
painful — you're importing tested functions, not copy-pasting notebook cells.

## Setup
```bash
pip install -r requirements.txt
```

## Dataset
Primary: `cnamuangtoun/resume-job-description-fit` (Hugging Face)
