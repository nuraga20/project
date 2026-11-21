# Before submitting by 21 November 2026

The computational project is complete. These account/identity and individual-defense steps remain the student's responsibility.

1. Review `README.md`, the annotation policy, the dataset, the error analysis, and the defense guide. Independently check the Kazakh judgments if possible. Any changed dataset must be versioned with newly reported metrics; do not retain old claims after changing labels.
2. Fill in your exact full name and student ID below. Describe your actual contribution after reviewing/modifying this package; keep the AI-assistance disclosure.
3. Create a GitHub repository and upload the **contents** of `kazakh_profanity_project`, including `artifacts/model.joblib`, all data files, results, and the notebook. Do not upload only the ZIP. Do not commit `.venv`, credentials, or access tokens. All required artifacts are small enough to include directly.
4. Check that the instructor and TAs can access the repository. A public repository is simplest if you are comfortable sharing; otherwise grant the required people access. Open `artifacts/model.joblib` in the repository and verify that it is downloadable.
5. Run one custom prediction from a fresh environment. `python -m unittest discover -s tests -v` should pass 16 tests. A reviewer should be able to predict without retraining.
6. Send the repository link directly to the instructor on Teams with your full name and student ID. This package has not sent any Teams message or published any GitHub repository.

## Submission message template

```text
Hello,

Please find my DSSC5113 bonus project: Kazakh Profanity Detection.

Full name: [YOUR EXACT FULL NAME]
Student ID: [YOUR STUDENT ID]
GitHub repository: [YOUR REPOSITORY LINK]

The repository contains the labeled dataset, fixed splits, training code,
saved model, inference instructions, test results, and AI-assistance disclosure.

Thank you.
```

## Actual student contribution (fill honestly)

```text
I personally reviewed / ran / changed: [describe what you actually did].
AI assistance included: draft implementation, AI annotations, synthetic examples,
training execution, evaluation, and draft documentation.
```
