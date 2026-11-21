# Annotation policy v1.0

## Target

Input: one Kazakh review/comment, possibly code-switched with Russian or written in informal transliteration. Output 1 if an identifiable **Kazakh obscene expression** is present, otherwise 0. This is a lexical profanity-presence task, not sentiment, general toxicity, personal abuse, threat detection, or a complete moderation system.

This project uses a deliberately narrow operational definition. It is a defensible project policy, not an authoritative dictionary of Kazakh speech. Language judgments should be checked by a competent Kazakh speaker before deployment.

## Positive class

- Explicit Kazakh vulgar sexual expressions and recognizable inflections/variants: e.g. `қотақ`, `қотағым`, `қотақбас`, `котак`, `котагым`, `qotak`, `qotaq`, obscene forms of `сігу`/`сіку`, and vulgar uses of `көт` and its inflections.
- Recognizable obscene phrases referring to the speaker's/interlocutor's genitals or sexual acts.
- `боқ` and inflections when used as a vulgar expletive or insult; this policy includes scatological profanity.
- Explicit sexual slurs such as `жәлеп` in an insulting/quoted context.
- Established obscene abbreviations `шшс` and `щщс` when used as standalone expletives, including positive-tone comments.
- Recognizable obfuscation (`қ*тақ`, `қо*ақ`), misspellings, and informal Latin variants. Do not guess from a string that is not identifiable.
- Quoted, reported, or negated profanity remains positive. “Do not write [obscene word]” still contains the word.

## Negative class

- Complaints, anger, negative product sentiment, multiple exclamation marks, and insulting tone alone.
- Ordinary insults such as `ақымақ`, `мал`, `есек`, `өтірікші`, or descriptions such as `қоқыс` without an obscene expression under this policy. They may still violate a platform's rules.
- Russian-only profanity with no identifiable Kazakh profanity. This follows the literal task scope; label 0 is NOT an assertion that the text is safe.
- Harmless substrings: `тамаша`, `аман`, `аманат`, `кісіге`, `есігі`, `тесігі`, `көтеру`. A short sequence that resembles an obscene word is not sufficient.
- Fully hidden words such as `****` with no recoverable obscene expression. We label visible evidence rather than guessing the hidden original.
- Merely saying “балағат сөз” does not contain a specific obscene expression.

## Uncertainty and exclusions

Review ambiguous abbreviation-like strings contextually. The candidate `шшСәлеметсіз` was excluded before training because treating its leading characters as an obscene abbreviation is uncertain. Identifying-person/handle contexts and a long numeric string were also excluded before splitting. Excluded source IDs and reasons are in `data/exclusions.csv`. No examples were removed because of model errors.

## Who annotated the dataset?

All new labels were produced by the ChatGPT/Codex assistant in this session after reading the selected comments. There was no independent human annotation, second annotator, or agreement study. Source sentiment labels were used only to enrich sampling for complaints, never as profanity ground truth. A profanity regex only selected candidates. One independently inspected nonmatching review (`pla120419pla`) contained `КОТАГЫМДЫ` and was assigned 1 despite the regex miss.

These are **AI annotations**, not human-verified gold labels. The exact serving model identifier was not available. The generation/annotation date was 20 November 2026. All 600 constructed examples are explicitly marked synthetic. There are 192 originally authored examples and 408 template/substitution additions. Both labels and every product substitution from a template remain in the same split.

## Recommended improvement

Have the student or another competent Kazakh speaker independently check all 2,000 annotations against this frozen policy, resolving ambiguous boundaries before a new experiment. Record any changed labels and create a new dataset/model version. Do not silently change the current labels and retain the current test metrics.

## Expansion annotation record

Version 2 retains the original 384 real judgments and adds 1,016 after contextual review of 1,110 candidates. `data/annotation_decisions_v2.csv` records the added labels and reasons. Four nonmatching comments containing vulgar spelling variants were labeled 1 after contextual inspection; matching was a selection tool, not ground truth. Newly identifying/ambiguous contexts were excluded before training. The old test IDs remain test, and new-test metrics exclude every original example.
