# Held-out error analysis

Model and threshold were frozen before this inspection. Test: 83 examples, 2 false positives, 3 false negatives. Threshold: 0.45. These cases are from the saved `test_predictions.csv`, not invented failures. All reference labels are AI annotations under the project's policy, not independent human gold labels.

| ID | Short excerpt | True | Predicted | Probability | Type |
|---|---|---:|---:|---:|---|
| real_fli001651fli | ... басын бастап, артынан видеодан көріңіздер ... | 0 | 1 | 0.450607 | False positive |
| real_pla059782pla | оте әдемі лайк бас ♥️♥️♥️ | 0 | 1 | 0.488757 | False positive |
| real_pla033326pla | ... Көтінті айрып жеринба ... | 1 | 0 | 0.351146 | False negative |
| real_pla129261pla | Көтіс ки екен маған унад | 1 | 0 | 0.393460 | False negative |
| synthetic_085_1 | Кремді жаққаннан кейін бетім ашыды, боқ. | 1 | 0 | 0.366349 | False negative |

## 1. Clean book review near the threshold

The book review says the content is incomplete and refers to a video. There is no identifiable Kazakh profanity under the policy. Its score barely crosses 0.45. The linear explanation shows positive contributions from `бас` (+0.0510), `бо` (+0.0353), and other short fragments. This supports an explanation of fragment overlap rather than understanding of obscene word boundaries. These overlapping features do not prove a single word caused the mistake.

Improvement: more naturally occurring clean examples with these fragments and word/context features in a future experiment. Raising the threshold might change this prediction but must be selected using validation, not this observed test error.

## 2. “Press like” misread as profane

`лайк бас` means “press like.” The clean positive-tone comment was flagged. The model assigns positive contributions to `от` (+0.1518), `бас` (+0.0814), `ас ` (+0.0794), and `бас ` (+0.0726). `бас` also occurs in profane compounds. This illustrates why character classifiers can confuse harmless overlapping fragments with profanity, especially with a small training set.

Improvement: independently collected clean imperative uses of `бас` and other hard negatives in the training pool. Do not silently move this test review into training while keeping the same benchmark claim.

## 3. Misspelling inside a long noisy review

The reference label is positive because the review contains a recognizable vulgar phrase around `Көтінті`. It also contains a lengthy unrelated quotation and spelling noise. L2-normalized TF-IDF distributes weight across many features. The local obscene variant is less directly represented than common training word families. This is a plausible explanation, not a proven causal diagnosis.

Improvement: more independently labeled spelling variants and diverse long-context examples. A future model could combine word and character features or use a contextual encoder. Label boundaries should also be checked by a Kazakh speaker.

## 4. Informal merged vulgar variant

`Көтіс ки` is annotated as a recognizable vulgar informal form in this policy. The model gives `көт` and `көті` positive contributions (+0.1050 and +0.0708), but they do not outweigh the intercept and other features. The phrase is short, noisy, and uncommon.

Improvement: more real variants of this word family, with negative examples that contain harmless `көтеру`-like fragments. Because this form is nonstandard, independent human verification would improve confidence in the reference label as well.

## 5. Scatological expletive in an otherwise ordinary complaint

The synthetic cream review is positive because `боқ` is included in the project's profanity policy. The model has learned positive weights for `боқ` (+0.0956) and ` боқ` (+0.0896), but the remaining sentence and intercept keep the total score below 0.45. The examples of this term in the small training set are less extensive than the dominant sexual word family.

Improvement: better-balanced lexical coverage of the policy's profanity families. A future rule-plus-model design might help literal presence detection, but it must be trained/validated and evaluated as a new system rather than patched using this test set.

## Implications for moderation

False positives can hide a customer's legitimate complaint, including the “press like” example. False negatives leave obscene language visible. A review queue can tolerate different costs than automatic deletion. The current probability is not a calibrated measure of real-world risk. The test has only 32 positive examples, so moving a few predictions substantially changes recall/F1.

The mixed benchmark F1 (0.9206) exceeds the real-only F1 (0.8889); the constructed contrast examples are easier. No real marketplace positive is present in the test, so goods-review profanity recall remains unmeasured. These are limitations of the evaluation even though its split and computation are correct.
