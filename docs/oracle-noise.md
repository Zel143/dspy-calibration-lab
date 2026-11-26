# The oracle-noise test

**Question:** if an LLM is given text that has no right answer, does it still report high confidence?

## Why

TempleOS, Terry A. Davis's operating system, shipped an "oracle" that drew random words and presented them as God speaking. People reading the output kept finding meaning in it. That is apophenia: seeing a pattern in noise.

The sister repo `prophetic-register-classifier` already asked this of its stylometric classifier (`src/null_control.py`). That classifier labels 23–37% of pure noise "prophetic", so its transfer results have to be read against that floor, not against 0%.

An LLM adds one thing the classifier doesn't have: a stated confidence. That makes noise a clean test for the Overtrust Index. Noise has no correct reading, so any confident reading of it is overtrust by construction. You don't need a judge or a gold label to see the miscalibration.

The study measures how a model reports confidence about textual form. It makes no claim about whether any text is true, inspired or Scripture.

## Design

`scripts/build_oracle_noise_dataset.py` draws 200 items, 40 for each of five conditions. The noise items are taken from the classifier repo's own null-control chunks, so the LLM and the classifier are scored on exactly the same text.

| condition | what it is | right answer exists? |
|---|---|---|
| `word_salad_uniform` | random KJV word types, the closest match to the TempleOS oracle | no |
| `word_salad_freq` | random KJV tokens, weighted by frequency | no |
| `verse_shuffle` | real KJV verses with the word order shuffled | no |
| `verse_flat` | the same verses as `verse_shuffle`, unshuffled (paired by chunk id) | meaningful; register unlabeled |
| `real_verse` | labeled seed-set verses, the positive control | yes |

Every item has the same surface form: lowercase and no punctuation. That way the model can't spot noise from formatting and has to read the words.

The same 200 items, in the same order, are asked under four framings, so every comparison between framings is paired:

| file | framing | gold on noise |
|---|---|---|
| `oracle_forced.jsonl` | prophetic / narrative / law-wisdom, no way out | none exists, and calibrated confidence is ≤ 1/3 |
| `oracle_abstain.jsonl` | the same question with `none` allowed | `none` |
| `oracle_detect_neutral.jsonl` | "meaningful or noise?" | `noise` |
| `oracle_detect_primed.jsonl` | the same question, but the text is introduced as an oracle's message | `noise` |

The main measures are:

- **Overtrust gap** (forced framing): mean confidence on noise minus 1/3.
- **Confident apophenia** (detect framings): the share of noise items called meaningful at confidence ≥ 0.8.
- **Priming effect:** how much the meaningful rate on the same noise items changes between the neutral and primed framings, tested with an exact McNemar test.
- **Order sensitivity:** how often a shuffled verse gets the same label as its unshuffled twin. The classifier gets 85.3% here, meaning it mostly ignores word order.

## Running it

The datasets are already committed. Only rebuild them if you change the sample:

```bash
python scripts/build_oracle_noise_dataset.py --classifier-repo ../prophetic-register-classifier
```

Run all four framings with the same model and strategy (800 calls at the defaults):

```bash
M=gemini/gemma-4-31b-it
python examples/run_basic.py --model $M --data data/oracle_forced.jsonl         --classes prophetic,narrative,law-wisdom
python examples/run_basic.py --model $M --data data/oracle_abstain.jsonl        --classes prophetic,narrative,law-wisdom,none
python examples/run_basic.py --model $M --data data/oracle_detect_neutral.jsonl --classes meaningful,noise
python examples/run_basic.py --model $M --data data/oracle_detect_primed.jsonl  --classes meaningful,noise
python scripts/analyze_oracle_noise.py --model $M --strategy direct
```

On Windows PowerShell, use `$M = "gemini/gemma-4-31b-it"`.

You can repeat the runs with `--strategy reasoned` to see whether chain-of-thought makes the model more willing to call noise noise, or just better at explaining it. The analysis script writes `results/oracle_noise_report.md`.

**Note:** the accuracy, Brier and ECE that `run_basic.py` prints for `oracle_forced` mean nothing, because noise has no right answer there. Use the analysis report instead.

## Caveats

- 40 items per condition is enough to see large effects, such as an overtrust gap of +0.4 or a lopsided priming split. It is too few for fine distinctions.
- The model can't refuse in `forced` by design. `abstain` is the fair framing for asking whether it would refuse if allowed.
- Shuffled verses keep the KJV vocabulary. A model that calls them "meaningful" may be responding to familiar words rather than imagining a message. The gap between `verse_shuffle` and `word_salad_uniform` helps separate the two.
- Results are per model and per strategy. Don't generalize from one model to LLMs in general.
