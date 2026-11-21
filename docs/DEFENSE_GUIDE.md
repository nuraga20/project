# Individual defense preparation

Read this alongside the source code. Explain the implementation in your own words; do not memorize claims you cannot demonstrate. The instructor may ask you to modify a setting or enter new text.

## A short honest opening

“I built a binary Kazakh profanity-presence classifier using character TF-IDF and Logistic Regression. The dataset contains 1,400 real reviews annotated by AI and 600 explicitly synthetic examples. AI assisted with implementation, annotation, and documentation; these labels have not been independently human-verified. Training, validation, and test are separated by related-review groups. The model was selected on validation and then evaluated on the held-out test. Its new-test F1 is 0.8571 (205 examples), while new real-only F1 is 0.9565 (145 examples), and physical-goods profanity recall is not established.”

Update the authorship statement to describe your own actual review, changes, and experiments. Do not claim that the original AI-authored work or AI annotations were your independent human work.

## Trace one comment

Run:

```bash
python -m src.predict --text "Тауар сапасы жақсы." --explain
```

1. Raw text enters `predict` in `src/predict.py`.
2. `normalize` in `src/common.py` uses NFKC, case folding, contact masking, and whitespace normalization. It preserves Kazakh letters and masking punctuation.
3. Saved `TfidfVectorizer` extracts character sequences of lengths 2, 3, 4, and 5. Only its training vocabulary is used; IDF is not refitted.
4. A sparse vector x is created. Each feature is a normalized TF-IDF value, not a language-model embedding.
5. Logistic Regression calculates `z=b+w·x`.
6. Sigmoid converts z into a model probability estimate `p=1/(1+exp(-z))`.
7. If p≥0.40, output 1; otherwise output 0.

The explanation lists the ten largest absolute local n-gram contributions. They add into z along with the omitted contributions and intercept. Because n-grams overlap, do not treat them as independent causal explanations.

## Expected questions

| Question | Key answer |
|---|---|
| Is a bad review necessarily profanity? | No. Sentiment and lexical profanity are different labels. |
| Why not use source ratings as labels? | A 1-star review can be completely clean; a positive review can contain profanity. |
| Why include synthetic positives with positive sentiment? | To reduce the shortcut “angry sentiment means profanity.” |
| How did you prevent duplicate leakage? | Exact deduplication, fixed-hash lexical similarity grouping, explicit synthetic parent groups, saved group-separated splits. |
| Why can hashing inspect all text for grouping? | This fixed transform learns no vocabulary/IDF and serves dataset deduplication only. Predictive TF-IDF is fitted only on training data. |
| What is an n-gram? | A contiguous sequence of characters; the 2-grams in `сапа` include `са`, `ап`, `па`. |
| Why character features? | Small data, informal spelling, Kazakh suffixes, and inspectable lightweight inference. |
| What does IDF do? | Downweights character sequences common across training documents; scikit-learn uses smoothed IDF by default. |
| What is sublinear TF? | Positive counts are transformed with `1+log(count)` before TF-IDF normalization. |
| Why regularization? | It discourages large coefficients that overfit limited data. |
| What does C do? | Larger C means weaker regularization; it was selected on validation. |
| Why balanced class weights? | This validation-selected option changes the relative contribution of the classes to logistic loss. |
| How were settings selected? | Six combinations of C/weighting and 13 thresholds, using validation class-1 F1. |
| Why threshold 0.40? | It was selected on validation; default 0.5 is not mandatory. |
| Did you refit after choosing the threshold? | No, the fitted model remained frozen so threshold and score distribution stayed matched. |
| How are unknown words handled? | They may share known n-grams; entirely unseen fragments are ignored. |
| How are long comments handled? | Full text is vectorized; normalization can dilute a short profane fragment within a long review. |
| What is a false positive? | A clean review is flagged; e.g. `лайк бас`. |
| What is a false negative? | A profane review is missed; there were 11 in the new primary test. |
| What are precision and recall here? | Precision 63/73: how many flags were correct. Recall 63/74: how many actual profane comments were found. |
| Why not just report accuracy? | It can be misleading when clean reviews dominate; the task asks for class-1 precision/recall/F1. |
| Is 0.86 F1 a production claim? | No. The benchmark is small, selected, partly synthetic, and AI-labeled. |
| What is the biggest domain limitation? | Real positives here come from app reviews, not physical marketplace goods. |
| Why not use GPT for all predictions? | The assignment requires training; this model is trained locally and saved for offline inference. |
| What if a review has Russian profanity only? | 0 under the documented narrow Kazakh-profanity policy; this is not complete content safety. |

## Small changes you should be able to demonstrate

- Change the text in the CLI or notebook and explain why its label follows the annotation policy.
- Show the real/synthetic split counts using pandas.
- Locate `ngram_range` and `C_grid` in `config.json`; explain what a change would do.
- Explain how lowering a threshold changes a prediction when p lies between the thresholds. Do not report a newly chosen threshold as a fresh held-out test experiment.
- Open `results/errors.csv` and explain at least three concrete mistakes.
- Load the saved model without running training.
- Explain what the source label meant and where the new profanity label is stored.

## Russian explanations for quick revision

- **Precision:** из всех отзывов, которые модель отметила как мат, какая доля действительно содержит казахский мат по нашей разметке.
- **Recall:** какую долю всех отзывов с матом модель нашла.
- **F1:** гармоническое среднее precision и recall.
- **Утечка:** почти одинаковые версии одного отзыва в train и test завышают оценку; поэтому связанные примеры держим вместе.
- **Синтетика:** примеры созданы ИИ, явно помечены; качество на них не равно качеству на реальных отзывах.
- **Разметка:** ИИ прочитал и разметил отобранные тексты, но независимой человеческой проверки не было.
- **Главное ограничение:** модель проверена на маленьком смешанном наборе; полноту поиска мата в реальных отзывах о физических товарах этот test не показывает.

## Explain the expansion honestly

The main dataset is exactly 2,000 unique normalized texts: 70% real (1,400), 30% synthetic (600). The source ratio is different from the profanity class ratio: 754 profane and 1,246 clean. Split sizes are 1,425 / 287 / 288. The original 83 test cases remain test, but their errors had already been inspected, so the 205 newly added test cases are the primary evaluation. New synthetic substitutions are grouped by all six products in a template, not just by individual pairs. Their correlated errors reduce mixed F1. More rows do not automatically mean more independent evidence or higher F1.
