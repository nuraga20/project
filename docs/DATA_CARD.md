# Data card — version 2.0

## Source and access

Real texts: **KazSAnDRA**, Rustem Yeshpanov and Huseyin Atakan Varol (ISSAI, Nazarbayev University), LREC-COLING 2024, pages 9657–9667. Repository: https://github.com/IS2AI/KazSAnDRA ; paper: https://aclanthology.org/2024.lrec-main.844/ . The authors state the dataset is **CC BY 4.0**. Retain attribution. They did not supply the new profanity annotations or endorse this classifier.

Pinned archive `dataset/06_sc_train_ib.zip`, member `06_sc_train_ib.csv`. The commit and SHA-256 are in `data/source_manifest.json`. `python -m src.download_source` verifies all 1,400 real selected texts against that pinned archive. Fixed annotated data are included, so training/inference do not require network access or the entire original corpus.

The original labels describe sentiment. They are never treated as profanity labels. The source's train archive serves only as a review pool for a new task; our evaluation is not an official KazSAnDRA sentiment result.

## Exact composition

| Origin | Label 0 | Label 1 | Total |
|---|---:|---:|---:|
| Real, AI-annotated | 946 | 454 | 1,400 |
| Synthetic | 300 | 300 | 600 |
| Total | 1,246 | 754 | 2,000 |

Thus exactly 70% real and 30% synthetic, by example count. Real domains: 655 physical-marketplace reviews and 745 app/digital-product reviews. Software reviews are relevant customer feedback but differ from physical-goods reviews. All real positive examples in this selected dataset come from the appstore domain. Constructed examples concern products, sellers, payment, delivery, or support.

## Selection and contextual AI annotation

V1 retained 384 real texts after the assistant inspected 400 candidates. V2 inspected 1,110 additional candidates: 350 lexicon-matching app reviews and 760 other comments. After contextual review it retained 324 matching reviews and 692 others, adding **1,016** real texts. Selection used fixed seeds 20261120, 20261121, and 20261122. The other candidate pool contained 500 marketplace and 260 app reviews, with half sampled from low source sentiment labels and half from the remaining pool. Selection order and exclusions determine the retained 692.

Added candidate lengths were 15–130 characters for matching app reviews and 20–125 for the other pool. These choices intentionally prioritize short comments and do not establish performance on all lengths. Regex matching selected candidates but did not supply the final labels. Four nonmatching added comments contained recognizable vulgar variants (`котендери`, `Сыгыл`, `ЖАЛАПТАР`, `шщс`) and were labeled positive after inspection. The original regex miss `КОТАГЫМДЫ` also remains positive. Source ratings are ignored for annotation.

The assistant read the candidate text in batches and recorded contextual decisions. New review decisions are in `data/annotation_decisions_v2.csv`; the complete frozen record is `base_annotations.csv`. Identifying-person/handle, non-Kazakh, unclear, or off-topic contexts were excluded before training; IDs/reasons are in `exclusions.csv`. Unused reviewed candidates were not added merely to increase size.

**All labels are AI annotations, not independently human-verified gold labels.** There was no second annotator, agreement study, or external paid annotation API. A Kazakh speaker should independently verify the policy and judgments. The exact serving model identifier was unavailable; authorship is recorded as ChatGPT/Codex assistant, 20 November 2026.

## Synthetic construction

- 192 original examples: 96 AI-authored contrast pairs.
- 408 additions: 34 AI-authored scenario templates, six product substitutions per template, and two labels per product.
- Total: 600 examples, 300 paired comparisons, **130 scenario families**, not 600 independent natural observations.

Both favorable and unfavorable sentiment appear among profane examples; clean complaints and ordinary insults are included as negatives. Some pairs differ by inserting one obscene expression, making them controlled lexical cases. They can be unnatural and do not reproduce the full distribution of real reviews. The 408 additions can be regenerated with `python -m src.build_synthetic` using `data/synthetic_scenarios.json`.

All products and both labels in a generated template share one group. Templates are never split by product substitution, so a nearly identical sentence with a different product cannot artificially inflate test performance. Some held-out template families proved difficult, producing correlated errors. Report that clustering rather than treating every row as independent evidence.

## Splits and expansion protocol

Seed 42. Train: 1,425; validation: 287; test: 288. Exact breakdown is in README and `dataset_summary.json`.

Preparation removes agreeing normalized exact duplicates and refuses conflicting labels. It unions original pair/template groups, punctuation-insensitive duplicate keys, and fixed-hash character cosine matches ≥0.90. Hashing learns no vocabulary/IDF; it is used only for grouping. Predictive TF-IDF learns vocabulary/IDF only from train. There are 1,529 groups; all 2,000 normalized texts are unique. Maximum cross-split lexical similarity under the documented hashing method is 0.8040. Semantic paraphrases below the threshold may still be missed.

The old split membership is saved in `previous_split_membership.csv`. Old test groups are pinned to test, old validation groups cannot move into train, and any merged group's holdout status takes priority. The 83 prior test examples were already inspected and are labeled `legacy_test`. **205 newly added test examples**, with no ID from any v1 split, are labeled `new_test` and form the primary v2 evaluation. A possible old-development example moved into test would be marked `previously_development` and excluded from the new-test score; none occurred here.

New test: 145 real (101 clean, 44 profane) + 60 synthetic (30 per class). All retained test: 200 real + 88 synthetic. Old test examples were not reused as training or validation examples.

## Privacy, limitations, and review

No reviewer profiles, names, or contact columns are distributed. Identifying contexts were excluded; selected text was checked for contact-like patterns. Runtime normalization masks email, URLs, @handles, and phone-like digit strings, but it does not reliably identify names.

This intentionally enriched sample does not estimate profanity prevalence. Finite-lexicon candidate selection biases positive examples toward known lexical families. Platform moderation may suppress profanity in marketplace reviews. Physical-goods test positives remain absent, so real physical-goods profanity recall is unmeasured. No claim is made about new slang, every dialect/transliteration, very long comments, or adversarial masking. Changes after reading the results need a newly untouched test set.
