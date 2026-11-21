# Error analysis — dataset/model version 2.0

The model was frozen at threshold **0.40** before inspecting test errors. No setting or label was changed to repair them. All reference labels are contextual AI judgments under the supplied policy, not independent human gold labels.

## Which test is being analyzed?

The **205 newly held-out examples** are the primary evaluation: TN=121, FP=10, FN=11, TP=63; precision 0.8630, recall 0.8514, F1 0.8571. Twenty-one mistakes occurred. The all-test file also contains 83 legacy examples whose errors were inspected in v1; all-test metrics are separately reported, with 25 errors among 288 examples.

## Concrete challenging cases

| ID | Short text/excerpt | True | Predicted | Probability | Cohort |
|---|---|---:|---:|---:|---|
| real_fli001801fli | Барлык отзывтарга косыламын. | 0 | 1 | 0.483847 | New |
| real_kas028029kas | Жұмсақ, трассада шу жоқ. | 0 | 1 | 0.416010 | New |
| real_pla036697pla | ... контакт тізімдер ... контактілер тізімін қайтарыңдар | 0 | 1 | 0.493237 | New |
| synthetic_v2_09_0_1 | Бұл телефон жарайды екен. Щщс, алдыңғы алғанымнан әлдеқайда жақсы жасалған. | 1 | 0 | 0.154457 | New |
| synthetic_v2_11_0_1 | Тауарды тексеретін жерде телефон жоқ дейді, ал төлем жасалып қойған, qotaq! | 1 | 0 | 0.362242 | New |
| synthetic_v2_23_0_0 | Бұл телефон үшін чек сақтадым, гарантия бойынша жауап күтемін. | 0 | 1 | 0.405713 | New |

### 1. Harmless overlapping fragments

The clean agreement comment was flagged. Local positive contributions include `от` (+0.2030), `ко` (+0.1439), `та` (+0.1103), and ` ко` (+0.1077). The model learned fragments from many profane words, which also occur in harmless text. This is evidence for a fragment-overlap explanation, not proof of one causal token. The short clean tire comment also crosses the threshold slightly.

Practical improvement: collect independent clean short reviews containing these fragments and add them to a future training version. Inspect word/character feature combinations. Select any new threshold using validation, not these test errors.

### 2. Code-switched “contacts” complaint

The comment about a contacts list is negative but contains no identifiable Kazakh profanity. Positive contributions include `так` (+0.1755), `ко` (+0.1587), `та` (+0.1507), and `ак` (+0.1165). Fragments in `контакт` overlap with frequent profanity spellings. A character model lacks a strong understanding of whole-word boundaries and context.

Practical improvement: more contextual hard negatives involving contact lists and similar harmless words, or word-boundary/context features in a new experiment.

### 3. Obscene abbreviation inside favorable sentiment

The `Щщс` example is positive under the policy even though the review praises the product. The model recognizes `щщс` fragments with positive contributions, but the intercept and remaining features keep the score low. All six products from template family 09 were held out together; their six positive variants were missed. They are **six correlated errors from one scenario**, not six unrelated failures.

Practical improvement: expand the training pool with independently written favorable comments containing obscene abbreviations, without reusing this observed test template. A literal-presence task may benefit from a validated lexical component, but that would be a new system requiring fresh evaluation.

### 4. Latin profanity in a longer complaint

The `qotaq` example is clearly marked synthetic and positive. Its score (0.3622) falls below 0.40. Some Latin fragments contribute positively, but ordinary complaint context outweighs them. Five of the six products from template family 11 are missed; the sixth crosses the threshold. This illustrates context/normalization dilution and the fragility of a fixed decision boundary for rare informal forms.

Practical improvement: broader real transliteration coverage and independently written contexts; evaluate on a new held-out sample. Do not claim all Latin spellings are solved by character features.

### 5. Clean warranty scenario falsely flagged

All six clean variants of template family 23 were flagged, with probabilities around 0.41–0.43. The phone variant has small positive contributions from `та`, `бо`, and ` бо` mixed with negative contributions. No single obscene expression is present. Again the family-level repetition produces correlated errors.

Practical improvement: broaden clean warranty/support examples. A fixed fragment-based model can learn accidental background associations; generated data should not be assumed easier or representative.

## Legacy mistakes and interpretation

The old noisy `Көтінті ...` phrase and informal `Көтіс ки ...` remain false negatives. The clean `лайк бас` remains a false positive. A legacy `боқ` cream example is still missed. These belong to the previously inspected legacy cohort, not the primary new-test claim.

New real-only F1 is 0.9565 (145 examples, 44 positives), while new synthetic errors are concentrated in unseen template families. All-test real-only F1 is 0.9449; all-test synthetic F1 is 0.7805. The lower mixed score after expansion does not justify tuning on this test. It reveals weaknesses that should inform a **future experiment with a newly untouched evaluation set**.

For moderation, false positives can suppress legitimate complaints and false negatives allow profanity. The probabilities are uncalibrated and the reference labels themselves need independent Kazakh-language verification. Physical-marketplace test examples are all clean, so physical-goods profanity recall remains unknown.
