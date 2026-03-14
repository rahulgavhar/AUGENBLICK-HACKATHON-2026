# Task 1 — What happens when I tokenize?

Model used: **BPE**

Mantra:

> ॐ भूर्भुवः स्व: तत्सवितुर्वरेण्यं भर्गो देवस्य धीमहि धियो यो नः प्रचोदयात् ॥

---

## Setup + training

Commands I ran:

```powershell
# (venv already activated)
New-Item -ItemType Directory -Force artifacts, data

# small corpus
'hello world
नमस्ते दुनिया
hello नमस्ते world' | Set-Content -Encoding utf8 data\corpus.txt

# train BPE
abctokz train --corpus data\corpus.txt --model bpe --vocab-size 200 --output artifacts\task1_bpe
```

Artifacts created:
- `artifacts/task1_bpe/vocab.json`
- `artifacts/task1_bpe/merges.txt`
- `artifacts/task1_bpe/manifest.json`

---

## Encode output (tokens + IDs)

Command:

```powershell
abctokz encode --model artifacts\task1_bpe --input data\mantra.txt
```

```text
Encoding: ॐ भूर्भुवः स्व:तत्सवितुर्वरेण्यं भर्गो देवस्य धीमहि धियो यो...     
┌─────┬───────┬────┐
│ Pos │ Token │ ID │
├─────┼───────┼────┤
│   0 │ ॐ    │  0 │
│   1 │ ##    │  0 │
│   2 │ ##भ   │  0 │
│   3 │ ##ू   │  0 │
│   4 │ ##र   │  0 │
│   5 │ ##्   │ 16 │
│   6 │ ##भ   │  0 │
│   7 │ ##ु   │  0 │
│   8 │ ##व   │  0 │
│   9 │ ##ः   │  0 │
│  10 │ ##    │  0 │
│  11 │ ##स   │ 14 │
│  12 │ ##्   │ 16 │
│  13 │ ##व   │  0 │
│  14 │ ##:   │  0 │
│  15 │ ##    │  0 │
│  16 │ ##त   │  8 │
│  17 │ ##त   │  8 │
│  18 │ ##्   │ 16 │
│  19 │ ##स   │ 14 │
│  20 │ ##व   │  0 │
│  21 │ ##ि   │  0 │
│  22 │ ##त   │  8 │
│  23 │ ##ु   │  0 │
│  24 │ ##र   │  0 │
│  25 │ ##्   │ 16 │
│  26 │ ##व   │  0 │
│  27 │ ##र   │  0 │
│  28 │ ##े   │ 15 │
│  29 │ ##ण   │  0 │
│  30 │ ##्   │ 16 │
│  31 │ ##य   │  0 │
│  32 │ ##ं   │  0 │
│  33 │ ##    │  0 │
│  34 │ ##भ   │  0 │
│  35 │ ##र   │  0 │
│  36 │ ##्   │ 16 │
│  37 │ ##ग   │  0 │
│  38 │ ##ो   │  0 │
│  39 │ ##    │  0 │
│  40 │ ##द   │  0 │
│  41 │ ##े   │ 15 │
│  42 │ ##व   │  0 │
│  43 │ ##स   │ 14 │
│  44 │ ##्   │ 16 │
│  45 │ ##य   │  0 │
│  46 │ ##    │  0 │
│  47 │ ##ध   │  0 │
│  48 │ ##ी   │  0 │
│  49 │ ##म   │ 10 │
│  50 │ ##ह   │  0 │
│  51 │ ##ि   │  0 │
│  52 │ ##    │  0 │
│  53 │ ##ध   │  0 │
│  54 │ ##ि   │  0 │
│  55 │ ##य   │  0 │
│  56 │ ##ो   │  0 │
│  57 │ ##    │  0 │
│  58 │ ##य   │  0 │
│  59 │ ##ो   │  0 │
│  60 │ ##    │  0 │
│  61 │ ##न   │  0 │
│  62 │ ##ः   │  0 │
│  63 │ ##    │  0 │
│  64 │ ##प   │  0 │
│  65 │ ##्   │ 16 │
│  66 │ ##र   │  0 │
│  67 │ ##च   │  0 │
│  68 │ ##ो   │  0 │
│  69 │ ##द   │  0 │
│  70 │ ##य   │  0 │
│  71 │ ##ा   │  0 │
│  72 │ ##त   │  8 │
│  73 │ ##्   │ 16 │
│  74 │ ##    │  0 │
│  75 │ ##॥   │  0 │
└─────┴───────┴────┘

```



---

## Trace of what happened from `encode()` → IDs

The encode pipeline in this repo is basically:

1. normalize (string → string)
2. pre-tokenize (string → list of pre-tokens)
3. model tokenize (each pre-token → subword pieces + IDs)
4. optional post-processing (e.g., BOS/EOS)

Then `decode()` does the inverse: IDs → token strings → text.

### Which files/classes were involved at each stage?

- Orchestrator: `src/abctokz/tokenizer.py`
  - class: `AugenblickTokenizer`
  - methods: `encode()` and `decode()`
- Normalizer stage:
  - factory: `src/abctokz/normalizers/__init__.py` → `build_normalizer(...)`
  - components: `DevanagariNormalizer`, `WhitespaceNormalizer`, chained via `SequenceNormalizer`
- Pre-tokenizer stage:
  - factory: `src/abctokz/pretokenizers/__init__.py` → `build_pretokenizer(...)`
  - component: `DevanagariAwarePreTokenizer` (inside `SequencePreTokenizer`)
- Model stage:
  - `src/abctokz/models/bpe.py` → `BPEModel.tokenize(...)`
  - merges/vocab loaded from `artifacts/task1_bpe/merges.txt` and `artifacts/task1_bpe/vocab.json`
- Decode stage:
  - `src/abctokz/tokenizer.py` → `decode()` (IDs → token strings)
  - `src/abctokz/decoders/subword_decoder.py` → `SubwordDecoder.decode(...)` (join pieces)

### What did the normalizer do to the string?

See section (3) for RAW vs NORMALIZED output.

In this config (from `bpe_multilingual`), normalization is a sequence:
- Devanagari-safe NFC normalization + exotic-space normalization
- then whitespace strip/collapse

Code:
- preset wiring: `src/abctokz/config/defaults.py`
- construction: `src/abctokz/normalizers/__init__.py`
- behavior: `src/abctokz/normalizers/devanagari.py` and `src/abctokz/normalizers/whitespace.py`

### What did the pre-tokenizer do after normalization?

See section (4) for the PRETOKENS list.

The Devanagari-aware pre-tokenizer:
- splits on whitespace
- can split inside a token when script changes (Devanagari ↔ Latin)
- keeps grapheme clusters intact (matras/halant stay attached)

Code:
- construction: `src/abctokz/pretokenizers/__init__.py`
- behavior: `src/abctokz/pretokenizers/devanagari_aware.py`

### How did the model turn pre-tokens into subword pieces and IDs?

`AugenblickTokenizer.encode()` loops over `pre_tokens` and calls `self._model.tokenize(pre_tok)`.

For BPE (`src/abctokz/models/bpe.py`):
- initialize pieces at character level (`_init_pieces`)
  - first char stays plain
  - later chars get `##` continuation prefix
- repeatedly apply the best-ranked merge pair until nothing matches (`_apply_merges`)
- map final piece strings to IDs using the vocab

Evidence artifacts:
- learned merge rules: `artifacts/task1_bpe/merges.txt`
- token→ID mapping: `artifacts/task1_bpe/vocab.json`

### How were pieces turned back into a string during `decode()`?

In `src/abctokz/tokenizer.py`, `decode()`:
- builds an inverse vocab (ID→token)
- optionally drops special tokens
- calls the decoder

`SubwordDecoder` (`src/abctokz/decoders/subword_decoder.py`) joins BPE pieces like this:
- token starts with `##` → strip `##` and glue to previous token
- otherwise → start a new word (insert a space before it, except at the beginning)

---

## 3) What the normalizer did

Command used to inspect it:

```powershell
python -c "from abctokz.config.defaults import bpe_multilingual; from abctokz.normalizers import build_normalizer; from abctokz.pretokenizers import build_pretokenizer; text=open('data/mantra.txt','r',encoding='utf-8').read().strip(); cfg=bpe_multilingual(vocab_size=200); n=build_normalizer(cfg.normalizer); p=build_pretokenizer(cfg.pretokenizer); norm=n.normalize(text); print('RAW:', text); print('NORMALIZED:', norm); print('PRETOKENS:', p.pre_tokenize(norm))"
```

Raw:
```text
ॐ भूर्भुवः स्व: तत्सवितुर्वरेण्यं भर्गो देवस्य धीमहि धियो यो नः प्रचोदयात् ॥
```

Normalized:
```text
 ॐ भूर्भुवः स्व: तत्सवितुर्वरेण्यं भर्गो देवस्य धीमहि धियो यो नः प्रचोदयात् ॥        'यो', 'नः', 'प्
```

---

## 4) What the pre-tokenizer did

Pre-tokens:
```text
['ॐ', 'भूर्भुवः', 'स्व:', 'तत्सवितुर्वरेण्यं', 'भर्गो', 'देवस्य', 'धीमहि', 'धियो', 'यो', 'नः', 'प्रचोदयात्', '॥']
```

Notes (what splits happened and why):
- 

---

## 5) How BPE turned pre-tokens into subword pieces

High-level description (short):
- Start with character-level pieces.
- Non-initial pieces use the `##` continuation prefix.
- Apply merge rules from `artifacts/task1_bpe/merges.txt` (ranked order).
- Map final pieces to IDs using `artifacts/task1_bpe/vocab.json`.

It happens in code:
- `src/abctokz/models/bpe.py`

---

## 6) How decode() reconstructs text

What decode does:
- IDs are mapped back to token strings using the inverse vocab.
- The decoder joins pieces:
  - if a piece starts with `##`, it’s glued to the previous piece (prefix removed)
  - otherwise it starts a new word (space inserted)

Where it happens in code:
- `src/abctokz/tokenizer.py` (`decode` builds token strings)
- `src/abctokz/decoders/subword_decoder.py` (joins `##` pieces)


---

## 7) Pipeline map (one paragraph)

When I call `encode(text)`, the tokenizer runs:
1) normalizer → 2) pre-tokenizer → 3) BPE model tokenization → 4) optional post-processing.
Then the `Encoding` object contains the final `tokens` and `ids`.


---



# Task 2 — Who Does What? Mapping Module Responsibilities

## 1. Responsibility → File/Module Mapping

### Training a tokenizer (learning vocabulary from text)

| Layer | File | Key symbol |
|---|---|---|
| CLI entry point / config ingestion | [src/abctokz/cli/train.py](../src/abctokz/cli/train.py#L18) | `train()` |
| Pipeline assembly + training call | [src/abctokz/tokenizer.py](../src/abctokz/tokenizer.py#L220) | `AugenblickTokenizer.from_config()` |
| In-place training dispatcher | [src/abctokz/tokenizer.py](../src/abctokz/tokenizer.py#L265) | `AugenblickTokenizer.train()` |
| Trainer factory | [src/abctokz/trainers/\_\_init\_\_.py](../src/abctokz/trainers/__init__.py#L11) | `build_trainer()` |
| BPE learning algorithm | [src/abctokz/trainers/bpe_trainer.py](../src/abctokz/trainers/bpe_trainer.py#L69) | `BPETrainer.train()` |
| Unigram learning algorithm | [src/abctokz/trainers/unigram_trainer.py](../src/abctokz/trainers/unigram_trainer.py#L87) | `UnigramTrainer.train()` |
| WordLevel learning algorithm | [src/abctokz/trainers/wordlevel_trainer.py](../src/abctokz/trainers/wordlevel_trainer.py#L19) | `WordLevelTrainer.train()` |

### Using a trained tokenizer to encode new text

| Layer | File | Key symbol |
|---|---|---|
| Full 4-stage encode pipeline | [src/abctokz/tokenizer.py](../src/abctokz/tokenizer.py#L93) | `AugenblickTokenizer.encode()` |
| Batch encoding | [src/abctokz/tokenizer.py](../src/abctokz/tokenizer.py#L158) | `AugenblickTokenizer.encode_batch()` |
| Decode path (ids → string) | [src/abctokz/tokenizer.py](../src/abctokz/tokenizer.py#L166) | `AugenblickTokenizer.decode()` |

### Saving and loading a tokenizer to/from disk

| Layer | File | Key symbol |
|---|---|---|
| Top-level artifact save | [src/abctokz/tokenizer.py](../src/abctokz/tokenizer.py#L313) | `AugenblickTokenizer.save()` |
| Top-level artifact load | [src/abctokz/tokenizer.py](../src/abctokz/tokenizer.py#L362) | `AugenblickTokenizer.load()` |
| BPE model-specific persistence | [src/abctokz/models/bpe.py](../src/abctokz/models/bpe.py#L132) | `BPEModel.save()` / `BPEModel.load()` |

### Measuring tokenizer quality (fertility, UNK rate, etc.)

| Layer | File | Key symbol |
|---|---|---|
| Core metric functions | [src/abctokz/eval/metrics.py](../src/abctokz/eval/metrics.py#L9) | `fertility()`, `unk_rate()`, `round_trip_success_rate()` |
| Single-tokenizer intrinsic eval | [src/abctokz/eval/intrinsic.py](../src/abctokz/eval/intrinsic.py#L17) | `evaluate_tokenizer()` |
| Multi-tokenizer timed benchmark | [src/abctokz/eval/benchmark.py](../src/abctokz/eval/benchmark.py#L30) | `BenchmarkRunner.run()` |
| Report formatting | [src/abctokz/eval/reports.py](../src/abctokz/eval/reports.py) | `results_to_markdown()` |

### Comparing abctokz against external tokenizers (HF / SentencePiece)

| Layer | File | Key symbol |
|---|---|---|
| Hugging Face adapter | [src/abctokz/adapters/hf.py](../src/abctokz/adapters/hf.py#L17) | `HFTokenizerAdapter` |
| SentencePiece adapter | [src/abctokz/adapters/sentencepiece.py](../src/abctokz/adapters/sentencepiece.py#L14) | `SentencePieceAdapter` |

**Why this separation exists (verified via runtime import inspection below):**
- `eval.metrics` imports only `abctokz.types` — it has zero dependency on training or model internals, so metrics can be computed against any tokenizer including external ones.
- `trainers.*` import `models.*` and `vocab.*` but never import `tokenizer` or `eval` — training logic cannot accidentally call inference or measurement code.
- `adapters.*` import only `abctokz.exceptions` and `abctokz.types` — external dependencies (HF, SP) are fully quarantined and cannot break core tokenizer code if packages are absent.

**Verified import graph (run `python -c "import importlib, inspect ..."`)**

```
[trainers.__init__]
  from abctokz.trainers.base import Trainer
  from abctokz.trainers.bpe_trainer import BPETrainer
  from abctokz.trainers.unigram_trainer import UnigramTrainer
  from abctokz.trainers.wordlevel_trainer import WordLevelTrainer
  from abctokz.config.schemas import BPETrainerConfig, TrainerConfig, ...

[trainers.bpe]
  from abctokz.config.schemas import BPETrainerConfig
  from abctokz.models.bpe import BPEModel        # only knows about its own model
  from abctokz.trainers.base import Trainer
  from abctokz.types import MergePair, MergeRules
  from abctokz.utils.seeds import set_seed

[eval.metrics]
  from abctokz.types import BenchmarkResult, Encoding  # no eval→train dependency

[adapters.hf]
  from abctokz.exceptions import AdapterError
  from abctokz.types import BenchmarkResult, Encoding  # no adapters→core dependency
```

---

## 2. One Especially Clean Module Boundary — The Trainer layer

**What it is:**  
[src/abctokz/trainers/base.py](../src/abctokz/trainers/base.py) defines a single abstract method `train(corpus) → Model`. Every concrete trainer implements exactly that contract and nothing else.

**Import discipline observed at runtime:**
```
[trainers.bpe]  imports:
  abctokz.config.schemas    ← config only
  abctokz.models.bpe        ← its own output model
  abctokz.trainers.base     ← its contract
  abctokz.vocab.*           ← low-level vocab primitives
  abctokz.utils.*           ← utilities (logging, seeds)
  # Does NOT import: tokenizer, eval, adapters, CLI
```

**Why it is satisfying:**
- The boundary enforces one rule: consume a corpus iterator, return a trained `Model` — no other side effects.
- Callers (`tokenizer.py`, `cli/train.py`) never need to know BPE or Unigram internals; they go through `build_trainer(config)` and receive a `Trainer`.
- Adding a fourth model family (e.g. WordPiece) requires creating `trainers/wordpiece_trainer.py` and one new branch in `build_trainer()` — nothing else changes.
- Because trainers only take iterators, they are easy to unit-test: pass a `["word1 word2", ...]` list, assert vocab output. No disk, no CLI needed.

**Verified with actual training run (`python examples/train_bpe.py`):**
```
Training BPE tokenizer...
  Vocabulary size: 310
  Saved to: C:\Users\...\bpe_tok

  Input:   'hello world'
  Tokens:  ['h', '##e', '##l', '##l', '##o', '## ', '##w', '##o', '##r', '##l', '##d']
  IDs:     [217, 50, 79, 79, 89, 0, 112, 89, 99, 79, 47]
  Decoded: 'helloworld'

  Input:   'नमस्ते दुनिया'
  Tokens:  ['न', '##म', '##स', '##्', '##त', '##े', '## ', '##द', '##ु', '##न', '##ि', '##य', '##ा']
  IDs:     [278, 156, 176, 201, 143, 191, 0, 145, 189, 147, 185, 159, 181]
  Decoded: 'नमस्तेदुनिया'
```

The BPE trainer produced its model correctly. The decoded output shows the separate issue discussed in Section 3 (spaces stripped post-decode — that is the load-gap problem, not a trainer problem).

---

## 3. One Blurry/Inconsistent Boundary — `save()` / `load()` drops the pipeline

**What the code promises:**  
`AugenblickTokenizer` is built with a full pipeline: normalizer → pretokenizer → model → post-processor → decoder. Every `encode()` call applies all stages in sequence ([tokenizer.py L93](../src/abctokz/tokenizer.py#L93)).

**What `save()` / `load()` actually does:**  
[`save()` at L313](../src/abctokz/tokenizer.py#L313) writes only `model_type` and `schema_version` to `config.json`. [`load()` at L362](../src/abctokz/tokenizer.py#L362) restores model + decoder — but **normalizer and pretokenizer are silently dropped**.

**Verified with code output:**

```
=== PIPELINE BEFORE SAVE ===
_normalizer   : <SequenceNormalizer object>
_pretokenizer : <SequencePreTokenizer object>
_post_processor: None
_decoder      : <SubwordDecoder object>

=== CONFIG.JSON ON DISK (what was actually saved) ===
{
  "model_type": "bpe",
  "schema_version": "1"
}

=== PIPELINE AFTER LOAD ===
_normalizer   : None          ← LOST
_pretokenizer : None          ← LOST
_post_processor: None
_decoder      : <SubwordDecoder object>
```

**Impact — encode outputs differ between original and loaded tokenizer:**

```
Text: 'hello world'
  original tokens : ['h', '##el', '##l', '##o', 'w', '##or', '##ld']   (7 tokens)
  loaded   tokens : ['h', '##e', '##l', '##l', '##o', '## ', '##w', '##o', '##r', '##l', '##d']  (11 tokens)
  Match: False

Text: 'नमस्ते दुनिया'
  original tokens : ['न', '##मस', '##्', '##ते', 'द', '##ु', '##नि', '##य', '##ा']  (9 tokens)
  loaded   tokens : ['न', '##म', '##स', '##्', '##त', '##े', '## ', '##द', '##ु', '##न', '##ि', '##य', '##ा']  (13 tokens)
  Match: False

Text: 'hello नमस्ते world'
  original tokens : ['h', '##el', '##l', '##o', 'न', '##मस', '##्', '##ते', 'w', '##or', '##ld']  (11 tokens)
  loaded   tokens : ['h', '##e', '##l', '##l', '##o', '## ', '##न', '##म', '##स', '##्', '##त', '##े', '## ', '##w', '##o', '##r', '##l', '##d']  (18 tokens)
  Match: False
```

All three Match values are `False`. The loaded tokenizer produces more tokens for every input because it lacks the pretokenizer that splits on word boundaries before BPE applies merges. This is a real behavioral divergence, not a cosmetic one.

**Why it is inconsistent:**  
The class-level docstring describes the pipeline as four cooperating stages. The constructor accepts normalizer and pretokenizer as first-class arguments. But `save/load` silently breaks the contract — a round-tripped tokenizer produces different token counts for the same input.

**Minimal, production-safe fix:**
1. In `save()`, dump the full `TokenizerConfig` (already serializable via Pydantic `.model_dump()`) instead of the current two-field dict.
2. In `load()`, read that config and call `build_normalizer()` / `build_pretokenizer()` to reconstruct the pipeline — the same functions already used in `from_config()` (L220).
3. Backward compat: if `config.json` lacks a `normalizer` key, silently skip reconstruction (old artifacts stay loadable).
4. Add one regression test: train a tokenizer with normalizer, save, load, assert `encode("hello world").tokens` is identical.


---




# Task 3 — The National Anthem Tokenization Test

Model Used:
**BPE (Byte Pair Encoding)**

---

# 1. Setup

## Input Data

To analyze tokenization behavior across scripts, we used the **first stanza of the Indian National Anthem (Jana Gana Mana)** in two different representations:

1. **English Transliteration (Latin Script)**
2. **Original Devanagari Script**

This allows us to observe how the same linguistic content behaves under different writing systems during tokenization.

---

# 2. Input Files

## 2.1 English Transliteration

**File:** `input_english_national_anthem.txt`

```
Jana Gana Mana Adhinayaka Jaya He
Bharata Bhagya Vidhata
Punjab Sindhu Gujarat Maratha
Dravida Utkala Banga
Vindhya Himachala Yamuna Ganga
Ucchala Jaladhi Taranga
Tava Shubha Name Jage
Tava Shubha Ashisha Mage
Gahe Tava Jaya Gatha
Jana Gana Mangala Dayaka Jaya He
Bharata Bhagya Vidhata
Jaya He Jaya He Jaya He
Jaya Jaya Jaya Jaya He
```

---

## 2.2 Devanagari Script

**File:** `input_devanagari_national_anthem.txt`

```
जन गण मन अधिनायक जय हे
भारत भाग्य विधाता
पंजाब सिंधु गुजरात मराठा
द्राविड़ उत्कल बंग
विंध्य हिमाचल यमुना गंगा
उच्छल जलधि तरंग
तव शुभ नामे जागे
तव शुभ आशीष मागे
गाहे तव जय गाथा
जन गण मंगलदायक जय हे
भारत भाग्य विधाता
जय हे जय हे जय हे
जय जय जय जय हे
```

---

# 3. Training the Tokenizer

The tokenizer was trained on **both scripts together** so that it could learn subword patterns from a **mixed multilingual corpus**.

```
abctokz train \
--corpus data/input_devanagari_national_anthem.txt \
--corpus data/input_english_national_anthem.txt \
--model bpe \
--vocab-size 300 \
--output artifacts/task3_anthem_bpe
```

### Training Output

```
Training bpe tokenizer...
Corpus: [input_devanagari_national_anthem.txt, input_english_national_anthem.txt]
Output: artifacts/task3_anthem_bpe

Done! Tokenizer saved.
Vocabulary size: 87
```

Because the dataset is extremely small, the tokenizer can only learn **very limited subword patterns**.

This becomes important later when we analyze the results.

---

# 4. Encoding the Text

After training, both input files were encoded using the trained tokenizer.

### Encoding Command

Encoding English Transliteration
```
abctokz encode --model artifacts/task3_anthem_bpe --input data/input_english_national_anthem.txt
```
output
```bash
Encoding: Jana Gana 
Mana Adhinayaka Jaya
         He
┏━━━━━┳━━━━━━━┳━━━━┓
┃ Pos ┃ Token ┃ ID ┃
┡━━━━━╇━━━━━━━╇━━━━┩
│   0 │ Ja    │ 63 │
│   1 │ ##n   │ 20 │
│   2 │ ##a   │  1 │
│   3 │ ##    │  0 │
│   4 │ ##G   │  0 │
│   5 │ ##a   │  1 │
│   6 │ ##n   │ 20 │
│   7 │ ##a   │  1 │
│   8 │ ##    │  0 │
│   9 │ ##M   │  0 │
│  10 │ ##a   │  1 │
│  11 │ ##n   │ 20 │
│  12 │ ##a   │  1 │
│  13 │ ##    │  0 │
│  14 │ ##A   │  0 │
│  15 │ ##d   │  8 │
│  16 │ ##h   │ 14 │
│  17 │ ##i   │ 18 │
│  18 │ ##n   │ 20 │
│  19 │ ##a   │  1 │
│  20 │ ##y   │ 28 │
│  21 │ ##a   │  1 │
│  22 │ ##k   │  0 │
│  23 │ ##a   │  1 │
│  24 │ ##    │  0 │
│  25 │ ##J   │  0 │
│  26 │ ##a   │  1 │
│  27 │ ##y   │ 28 │
│  28 │ ##a   │  1 │
│  29 │ ##    │  0 │
│  30 │ ##H   │  0 │
│  31 │ ##e   │ 11 │
└─────┴───────┴────┘
 Encoding: Bharata  
   Bhagya Vidhata   
┏━━━━━┳━━━━━━━┳━━━━┓
┃ Pos ┃ Token ┃ ID ┃
┡━━━━━╇━━━━━━━╇━━━━┩
│   0 │ B     │ 52 │
│   1 │ ##h   │ 14 │
│   2 │ ##a   │  1 │
│   3 │ ##r   │ 22 │
│   4 │ ##a   │  1 │
│   5 │ ##t   │ 24 │
│   6 │ ##a   │  1 │
│   7 │ ##    │  0 │
│   8 │ ##B   │  0 │
│   9 │ ##h   │ 14 │
│  10 │ ##a   │  1 │
│  11 │ ##g   │ 12 │
│  12 │ ##y   │ 28 │
│  13 │ ##a   │  1 │
│  14 │ ##    │  0 │
│  15 │ ##V   │  0 │
│  16 │ ##i   │ 18 │
│  17 │ ##d   │  8 │
│  18 │ ##h   │ 14 │
│  19 │ ##a   │  1 │
│  20 │ ##t   │ 24 │
│  21 │ ##a   │  1 │
└─────┴───────┴────┘
  Encoding: Punjab  
   Sindhu Gujarat   
      Maratha       
┏━━━━━┳━━━━━━━┳━━━━┓
┃ Pos ┃ Token ┃ ID ┃
┡━━━━━╇━━━━━━━╇━━━━┩
│   0 │ P     │  0 │
│   1 │ ##u   │ 26 │
│   2 │ ##n   │ 20 │
│   3 │ ##j   │  0 │
│   4 │ ##a   │  1 │
│   5 │ ##b   │  6 │
│   6 │ ##    │  0 │
│   7 │ ##S   │  0 │
│   8 │ ##i   │ 18 │
│   9 │ ##n   │ 20 │
│  10 │ ##d   │  8 │
│  11 │ ##h   │ 14 │
│  12 │ ##u   │ 26 │
│  13 │ ##    │  0 │
│  14 │ ##G   │  0 │
│  15 │ ##u   │ 26 │
│  16 │ ##j   │  0 │
│  17 │ ##a   │  1 │
│  18 │ ##r   │ 22 │
│  19 │ ##a   │  1 │
│  20 │ ##t   │ 24 │
│  21 │ ##    │  0 │
│  22 │ ##M   │  0 │
│  23 │ ##a   │  1 │
│  24 │ ##r   │ 22 │
│  25 │ ##a   │  1 │
│  26 │ ##t   │ 24 │
│  27 │ ##h   │ 14 │
│  28 │ ##a   │  1 │
└─────┴───────┴────┘
 Encoding: Dravida  
    Utkala Banga    
┏━━━━━┳━━━━━━━┳━━━━┓
┃ Pos ┃ Token ┃ ID ┃
┡━━━━━╇━━━━━━━╇━━━━┩
│   0 │ D     │  0 │
│   1 │ ##r   │ 22 │
│   2 │ ##a   │  1 │
│   3 │ ##v   │ 27 │
│   4 │ ##i   │ 18 │
│   5 │ ##d   │  8 │
│   6 │ ##a   │  1 │
│   7 │ ##    │  0 │
│   8 │ ##U   │  0 │
│   9 │ ##t   │ 24 │
│  10 │ ##k   │  0 │
│  11 │ ##a   │  1 │
│  12 │ ##l   │  0 │
│  13 │ ##a   │  1 │
│  14 │ ##    │  0 │
│  15 │ ##B   │  0 │
│  16 │ ##a   │  1 │
│  17 │ ##n   │ 20 │
│  18 │ ##g   │ 12 │
│  19 │ ##a   │  1 │
└─────┴───────┴────┘
 Encoding: Vindhya  
  Himachala Yamuna  
       Ganga        
┏━━━━━┳━━━━━━━┳━━━━┓
┃ Pos ┃ Token ┃ ID ┃
┡━━━━━╇━━━━━━━╇━━━━┩
│   0 │ V     │ 68 │
│   1 │ ##i   │ 18 │
│   2 │ ##n   │ 20 │
│   3 │ ##d   │  8 │
│   4 │ ##h   │ 14 │
│   5 │ ##y   │ 28 │
│   6 │ ##a   │  1 │
│   7 │ ##    │  0 │
│   8 │ ##H   │  0 │
│   9 │ ##i   │ 18 │
│  10 │ ##m   │  0 │
│  11 │ ##a   │  1 │
│  12 │ ##c   │  0 │
│  13 │ ##h   │ 14 │
│  14 │ ##a   │  1 │
│  15 │ ##l   │  0 │
│  16 │ ##a   │  1 │
│  17 │ ##    │  0 │
│  18 │ ##Y   │  0 │
│  19 │ ##a   │  1 │
│  20 │ ##m   │  0 │
│  21 │ ##u   │ 26 │
│  22 │ ##n   │ 20 │
│  23 │ ##a   │  1 │
│  24 │ ##    │  0 │
│  25 │ ##G   │  0 │
│  26 │ ##a   │  1 │
│  27 │ ##n   │ 20 │
│  28 │ ##g   │ 12 │
│  29 │ ##a   │  1 │
└─────┴───────┴────┘
 Encoding: Ucchala  
  Jaladhi Taranga   
┏━━━━━┳━━━━━━━┳━━━━┓
┃ Pos ┃ Token ┃ ID ┃
┡━━━━━╇━━━━━━━╇━━━━┩
│   0 │ U     │  0 │
│   1 │ ##c   │  0 │
│   2 │ ##c   │  0 │
│   3 │ ##h   │ 14 │
│   4 │ ##a   │  1 │
│   5 │ ##l   │  0 │
│   6 │ ##a   │  1 │
│   7 │ ##    │  0 │
│   8 │ ##J   │  0 │
│   9 │ ##a   │  1 │
│  10 │ ##l   │  0 │
│  11 │ ##a   │  1 │
│  12 │ ##d   │  8 │
│  13 │ ##h   │ 14 │
│  14 │ ##i   │ 18 │
│  15 │ ##    │  0 │
│  16 │ ##T   │  0 │
│  17 │ ##a   │  1 │
│  18 │ ##r   │ 22 │
│  19 │ ##a   │  1 │
│  20 │ ##n   │ 20 │
│  21 │ ##g   │ 12 │
│  22 │ ##a   │  1 │
└─────┴───────┴────┘
   Encoding: Tava   
  Shubha Name Jage  
┏━━━━━┳━━━━━━━┳━━━━┓
┃ Pos ┃ Token ┃ ID ┃
┡━━━━━╇━━━━━━━╇━━━━┩
│   0 │ T     │ 66 │
│   1 │ ##a   │  1 │
│   2 │ ##v   │ 27 │
│   3 │ ##a   │  1 │
│   4 │ ##    │  0 │
│   5 │ ##S   │  0 │
│   6 │ ##h   │ 14 │
│   7 │ ##u   │ 26 │
│   8 │ ##b   │  6 │
│   9 │ ##h   │ 14 │
│  10 │ ##a   │  1 │
│  11 │ ##    │  0 │
│  12 │ ##N   │  0 │
│  13 │ ##a   │  1 │
│  14 │ ##m   │  0 │
│  15 │ ##e   │ 11 │
│  16 │ ##    │  0 │
│  17 │ ##J   │  0 │
│  18 │ ##a   │  1 │
│  19 │ ##g   │ 12 │
│  20 │ ##e   │ 11 │
└─────┴───────┴────┘
   Encoding: Tava   
Shubha Ashisha Mage 
┏━━━━━┳━━━━━━━┳━━━━┓
┃ Pos ┃ Token ┃ ID ┃
┡━━━━━╇━━━━━━━╇━━━━┩
│   0 │ T     │ 66 │
│   1 │ ##a   │  1 │
│   2 │ ##v   │ 27 │
│   3 │ ##a   │  1 │
│   4 │ ##    │  0 │
│   5 │ ##S   │  0 │
│   6 │ ##h   │ 14 │
│   7 │ ##u   │ 26 │
│   8 │ ##b   │  6 │
│   9 │ ##h   │ 14 │
│  10 │ ##a   │  1 │
│  11 │ ##    │  0 │
│  12 │ ##A   │  0 │
│  13 │ ##s   │  0 │
│  14 │ ##h   │ 14 │
│  15 │ ##i   │ 18 │
│  16 │ ##s   │  0 │
│  17 │ ##h   │ 14 │
│  18 │ ##a   │  1 │
│  19 │ ##    │  0 │
│  20 │ ##M   │  0 │
│  21 │ ##a   │  1 │
│  22 │ ##g   │ 12 │
│  23 │ ##e   │ 11 │
└─────┴───────┴────┘
Encoding: Gahe Tava 
     Jaya Gatha     
┏━━━━━┳━━━━━━━┳━━━━┓
┃ Pos ┃ Token ┃ ID ┃
┡━━━━━╇━━━━━━━╇━━━━┩
│   0 │ G     │ 56 │
│   1 │ ##a   │  1 │
│   2 │ ##h   │ 14 │
│   3 │ ##e   │ 11 │
│   4 │ ##    │  0 │
│   5 │ ##T   │  0 │
│   6 │ ##a   │  1 │
│   7 │ ##v   │ 27 │
│   8 │ ##a   │  1 │
│   9 │ ##    │  0 │
│  10 │ ##J   │  0 │
│  11 │ ##a   │  1 │
│  12 │ ##y   │ 28 │
│  13 │ ##a   │  1 │
│  14 │ ##    │  0 │
│  15 │ ##G   │  0 │
│  16 │ ##a   │  1 │
│  17 │ ##t   │ 24 │
│  18 │ ##h   │ 14 │
│  19 │ ##a   │  1 │
└─────┴───────┴────┘
Encoding: Jana Gana 
Mangala Dayaka Jaya 
         He
┏━━━━━┳━━━━━━━┳━━━━┓
┃ Pos ┃ Token ┃ ID ┃
┡━━━━━╇━━━━━━━╇━━━━┩
│   0 │ Ja    │ 63 │
│   1 │ ##n   │ 20 │
│   2 │ ##a   │  1 │
│   3 │ ##    │  0 │
│   4 │ ##G   │  0 │
│   5 │ ##a   │  1 │
│   6 │ ##n   │ 20 │
│   7 │ ##a   │  1 │
│   8 │ ##    │  0 │
│   9 │ ##M   │  0 │
│  10 │ ##a   │  1 │
│  11 │ ##n   │ 20 │
│  12 │ ##g   │ 12 │
│  13 │ ##a   │  1 │
│  14 │ ##l   │  0 │
│  15 │ ##a   │  1 │
│  16 │ ##    │  0 │
│  17 │ ##D   │  0 │
│  18 │ ##a   │  1 │
│  19 │ ##y   │ 28 │
│  20 │ ##a   │  1 │
│  21 │ ##k   │  0 │
│  22 │ ##a   │  1 │
│  23 │ ##    │  0 │
│  24 │ ##J   │  0 │
│  25 │ ##a   │  1 │
│  26 │ ##y   │ 28 │
│  27 │ ##a   │  1 │
│  28 │ ##    │  0 │
│  29 │ ##H   │  0 │
│  30 │ ##e   │ 11 │
└─────┴───────┴────┘
 Encoding: Bharata  
   Bhagya Vidhata   
┏━━━━━┳━━━━━━━┳━━━━┓
┃ Pos ┃ Token ┃ ID ┃
┡━━━━━╇━━━━━━━╇━━━━┩
│   0 │ B     │ 52 │
│   1 │ ##h   │ 14 │
│   2 │ ##a   │  1 │
│   3 │ ##r   │ 22 │
│   4 │ ##a   │  1 │
│   5 │ ##t   │ 24 │
│   6 │ ##a   │  1 │
│   7 │ ##    │  0 │
│   8 │ ##B   │  0 │
│   9 │ ##h   │ 14 │
│  10 │ ##a   │  1 │
│  11 │ ##g   │ 12 │
│  12 │ ##y   │ 28 │
│  13 │ ##a   │  1 │
│  14 │ ##    │  0 │
│  15 │ ##V   │  0 │
│  16 │ ##i   │ 18 │
│  17 │ ##d   │  8 │
│  18 │ ##h   │ 14 │
│  19 │ ##a   │  1 │
│  20 │ ##t   │ 24 │
│  21 │ ##a   │  1 │
└─────┴───────┴────┘
 Encoding: Jaya He  
  Jaya He Jaya He   
┏━━━━━┳━━━━━━━┳━━━━┓
┃ Pos ┃ Token ┃ ID ┃
┡━━━━━╇━━━━━━━╇━━━━┩
│   0 │ Ja    │ 63 │
│   1 │ ##y   │ 28 │
│   2 │ ##a   │  1 │
│   3 │ ##    │  0 │
│   4 │ ##H   │  0 │
│   5 │ ##e   │ 11 │
│   6 │ ##    │  0 │
│   7 │ ##J   │  0 │
│   8 │ ##a   │  1 │
│   9 │ ##y   │ 28 │
│  10 │ ##a   │  1 │
│  11 │ ##    │  0 │
│  12 │ ##H   │  0 │
│  13 │ ##e   │ 11 │
│  14 │ ##    │  0 │
│  15 │ ##J   │  0 │
│  16 │ ##a   │  1 │
│  17 │ ##y   │ 28 │
│  18 │ ##a   │  1 │
│  19 │ ##    │  0 │
│  20 │ ##H   │  0 │
│  21 │ ##e   │ 11 │
└─────┴───────┴────┘
Encoding: Jaya Jaya 
    Jaya Jaya He    
┏━━━━━┳━━━━━━━┳━━━━┓
┃ Pos ┃ Token ┃ ID ┃
┡━━━━━╇━━━━━━━╇━━━━┩
│   0 │ Ja    │ 63 │
│   1 │ ##y   │ 28 │
│   2 │ ##a   │  1 │
│   3 │ ##    │  0 │
│   4 │ ##J   │  0 │
│   5 │ ##a   │  1 │
│   6 │ ##y   │ 28 │
│   7 │ ##a   │  1 │
│   8 │ ##    │  0 │
│   9 │ ##J   │  0 │
│  10 │ ##a   │  1 │
│  11 │ ##y   │ 28 │
│  12 │ ##a   │  1 │
│  13 │ ##    │  0 │
│  14 │ ##J   │  0 │
│  15 │ ##a   │  1 │
│  16 │ ##y   │ 28 │
│  17 │ ##a   │  1 │
│  18 │ ##    │  0 │
│  19 │ ##H   │  0 │
│  20 │ ##e   │ 11 │
└─────┴───────┴────┘
```
and

Encoding Devanagari scripts
```
abctokz encode --model artifacts/task3_anthem_bpe --input data/input_devanagari_national_anthem.txt
```
output
```bash
Encoding: जन गण मन 
     अधिनायक जय हे     
┏━━━━━┳━━━━━━━┳━━━━┓
┃ Pos ┃ Token ┃ ID ┃
┡━━━━━╇━━━━━━━╇━━━━┩
│   0 │ जन    │ 73 │
│   1 │ ##    │  0 │
│   2 │ ##ग   │ 30 │
│   3 │ ##ण   │ 33 │
│   4 │ ##    │  0 │
│   5 │ ##म   │  0 │
│   6 │ ##न   │ 39 │
│   7 │ ##    │  0 │
│   8 │ ##अ   │  0 │
│   9 │ ##ध   │ 36 │
│  10 │ ##ि    │ 46 │
│  11 │ ##न   │ 39 │
│  12 │ ##ा    │ 45 │
│  13 │ ##य   │ 41 │
│  14 │ ##क   │  0 │
│  15 │ ##    │  0 │
│  16 │ ##ज   │  0 │
│  17 │ ##य   │ 41 │
│  18 │ ##    │  0 │
│  19 │ ##ह   │  0 │
│  20 │ ##े    │ 50 │
└─────┴───────┴────┘
 Encoding: भारत भाग्य  
        विधाता
┏━━━━━┳━━━━━━━┳━━━━┓
┃ Pos ┃ Token ┃ ID ┃
┡━━━━━╇━━━━━━━╇━━━━┩
│   0 │ भा     │ 80 │
│   1 │ ##र   │ 42 │
│   2 │ ##त   │ 34 │
│   3 │ ##    │  0 │
│   4 │ ##भ   │ 40 │
│   5 │ ##ा    │ 45 │
│   6 │ ##ग   │ 30 │
│   7 │ ##्    │ 51 │
│   8 │ ##य   │ 41 │
│   9 │ ##    │  0 │
│  10 │ ##व   │ 44 │
│  11 │ ##ि    │ 46 │
│  12 │ ##ध   │ 36 │
│  13 │ ##ा    │ 45 │
│  14 │ ##त   │ 34 │
│  15 │ ##ा    │ 45 │
└─────┴───────┴────┘
  Encoding: पंजाब सिंधु  
      गुजरात मराठा      
┏━━━━━┳━━━━━━━┳━━━━┓
┃ Pos ┃ Token ┃ ID ┃
┡━━━━━╇━━━━━━━╇━━━━┩
│   0 │ प     │  0 │
│   1 │ ##ं    │  0 │
│   2 │ ##ज   │  0 │
│   3 │ ##ा    │ 45 │
│   4 │ ##ब   │  0 │
│   5 │ ##    │  0 │
│   6 │ ##स   │  0 │
│   7 │ ##ि    │ 46 │
│   8 │ ##ं    │  0 │
│   9 │ ##ध   │ 36 │
│  10 │ ##ु    │ 48 │
│  11 │ ##    │  0 │
│  12 │ ##ग   │ 30 │
│  13 │ ##ु    │ 48 │
│  14 │ ##ज   │  0 │
│  15 │ ##र   │ 42 │
│  16 │ ##ा    │ 45 │
│  17 │ ##त   │ 34 │
│  18 │ ##    │  0 │
│  19 │ ##म   │  0 │
│  20 │ ##र   │ 42 │
│  21 │ ##ा    │ 45 │
│  22 │ ##ठ   │  0 │
│  23 │ ##ा    │ 45 │
└─────┴───────┴────┘
Encoding: द्राविड़ उत्कल 
         बंग
┏━━━━━┳━━━━━━━┳━━━━┓
┃ Pos ┃ Token ┃ ID ┃
┡━━━━━╇━━━━━━━╇━━━━┩
│   0 │ द     │  0 │
│   1 │ ##्    │ 51 │
│   2 │ ##र   │ 42 │
│   3 │ ##ा    │ 45 │
│   4 │ ##व   │ 44 │
│   5 │ ##ि    │ 46 │
│   6 │ ##ड   │  0 │
│   7 │ ##़    │  0 │
│   8 │ ##    │  0 │
│   9 │ ##उ   │  0 │
│  10 │ ##त   │ 34 │
│  11 │ ##्    │ 51 │
│  12 │ ##क   │  0 │
│  13 │ ##ल   │  0 │
│  14 │ ##    │  0 │
│  15 │ ##ब   │  0 │
│  16 │ ##ं    │  0 │
│  17 │ ##ग   │ 30 │
└─────┴───────┴────┘
 Encoding: विंध्य हिमाचल 
       यमुना गंगा       
┏━━━━━┳━━━━━━━┳━━━━┓
┃ Pos ┃ Token ┃ ID ┃
┡━━━━━╇━━━━━━━╇━━━━┩
│   0 │ व     │ 81 │
│   1 │ ##ि    │ 46 │
│   2 │ ##ं    │  0 │
│   3 │ ##ध   │ 36 │
│   4 │ ##्    │ 51 │
│   5 │ ##य   │ 41 │
│   6 │ ##    │  0 │
│   7 │ ##ह   │  0 │
│   8 │ ##ि    │ 46 │
│   9 │ ##म   │  0 │
│  10 │ ##ा    │ 45 │
│  11 │ ##च   │  0 │
│  12 │ ##ल   │  0 │
│  13 │ ##    │  0 │
│  14 │ ##य   │ 41 │
│  15 │ ##म   │  0 │
│  16 │ ##ु    │ 48 │
│  17 │ ##न   │ 39 │
│  18 │ ##ा    │ 45 │
│  19 │ ##    │  0 │
│  20 │ ##ग   │ 30 │
│  21 │ ##ं    │  0 │
│  22 │ ##ग   │ 30 │
│  23 │ ##ा    │ 45 │
└─────┴───────┴────┘
 Encoding: उच्छल जलधि 
        तरंग
┏━━━━━┳━━━━━━━┳━━━━┓
┃ Pos ┃ Token ┃ ID ┃
┡━━━━━╇━━━━━━━╇━━━━┩
│   0 │ उ     │  0 │
│   1 │ ##च   │  0 │
│   2 │ ##्    │ 51 │
│   3 │ ##छ   │  0 │
│   4 │ ##ल   │  0 │
│   5 │ ##    │  0 │
│   6 │ ##ज   │  0 │
│   7 │ ##ल   │  0 │
│   8 │ ##ध   │ 36 │
│   9 │ ##ि    │ 46 │
│  10 │ ##    │  0 │
│  11 │ ##त   │ 34 │
│  12 │ ##र   │ 42 │
│  13 │ ##ं    │  0 │
│  14 │ ##ग   │ 30 │
└─────┴───────┴────┘
 Encoding: तव शुभ नामे 
         जागे
┏━━━━━┳━━━━━━━┳━━━━┓
┃ Pos ┃ Token ┃ ID ┃
┡━━━━━╇━━━━━━━╇━━━━┩
│   0 │ तव    │ 76 │
│   1 │ ##    │  0 │
│   2 │ ##श   │  0 │
│   3 │ ##ु    │ 48 │
│   4 │ ##भ   │ 40 │
│   5 │ ##    │  0 │
│   6 │ ##न   │ 39 │
│   7 │ ##ा    │ 45 │
│   8 │ ##म   │  0 │
│   9 │ ##े    │ 50 │
│  10 │ ##    │  0 │
│  11 │ ##ज   │  0 │
│  12 │ ##ा    │ 45 │
│  13 │ ##ग   │ 30 │
│  14 │ ##े    │ 50 │
└─────┴───────┴────┘
Encoding: तव शुभ आशीष 
         मागे
┏━━━━━┳━━━━━━━┳━━━━┓
┃ Pos ┃ Token ┃ ID ┃
┡━━━━━╇━━━━━━━╇━━━━┩
│   0 │ तव    │ 76 │
│   1 │ ##    │  0 │
│   2 │ ##श   │  0 │
│   3 │ ##ु    │ 48 │
│   4 │ ##भ   │ 40 │
│   5 │ ##    │  0 │
│   6 │ ##आ   │  0 │
│   7 │ ##श   │  0 │
│   8 │ ##ी    │  0 │
│   9 │ ##ष   │  0 │
│  10 │ ##    │  0 │
│  11 │ ##म   │  0 │
│  12 │ ##ा    │ 45 │
│  13 │ ##ग   │ 30 │
│  14 │ ##े    │ 50 │
└─────┴───────┴────┘
 Encoding: गाहे तव जय 
         गाथा
┏━━━━━┳━━━━━━━┳━━━━┓
┃ Pos ┃ Token ┃ ID ┃
┡━━━━━╇━━━━━━━╇━━━━┩
│   0 │ ग     │ 70 │
│   1 │ ##ा    │ 45 │
│   2 │ ##ह   │  0 │
│   3 │ ##े    │ 50 │
│   4 │ ##    │  0 │
│   5 │ ##त   │ 34 │
│   6 │ ##व   │ 44 │
│   7 │ ##    │  0 │
│   8 │ ##ज   │  0 │
│   9 │ ##य   │ 41 │
│  10 │ ##    │  0 │
│  11 │ ##ग   │ 30 │
│  12 │ ##ा    │ 45 │
│  13 │ ##थ   │  0 │
│  14 │ ##ा    │ 45 │
└─────┴───────┴────┘
  Encoding: जन गण   
    मंगलदायक जय हे     
┏━━━━━┳━━━━━━━┳━━━━┓
┃ Pos ┃ Token ┃ ID ┃
┡━━━━━╇━━━━━━━╇━━━━┩
│   0 │ जन    │ 73 │
│   1 │ ##    │  0 │
│   2 │ ##ग   │ 30 │
│   3 │ ##ण   │ 33 │
│   4 │ ##    │  0 │
│   5 │ ##म   │  0 │
│   6 │ ##ं    │  0 │
│   7 │ ##ग   │ 30 │
│   8 │ ##ल   │  0 │
│   9 │ ##द   │  0 │
│  10 │ ##ा    │ 45 │
│  11 │ ##य   │ 41 │
│  12 │ ##क   │  0 │
│  13 │ ##    │  0 │
│  14 │ ##ज   │  0 │
│  15 │ ##य   │ 41 │
│  16 │ ##    │  0 │
│  17 │ ##ह   │  0 │
│  18 │ ##े    │ 50 │
└─────┴───────┴────┘
 Encoding: भारत भाग्य  
        विधाता
┏━━━━━┳━━━━━━━┳━━━━┓
┃ Pos ┃ Token ┃ ID ┃
┡━━━━━╇━━━━━━━╇━━━━┩
│   0 │ भा     │ 80 │
│   1 │ ##र   │ 42 │
│   2 │ ##त   │ 34 │
│   3 │ ##    │  0 │
│   4 │ ##भ   │ 40 │
│   5 │ ##ा    │ 45 │
│   6 │ ##ग   │ 30 │
│   7 │ ##्    │ 51 │
│   8 │ ##य   │ 41 │
│   9 │ ##    │  0 │
│  10 │ ##व   │ 44 │
│  11 │ ##ि    │ 46 │
│  12 │ ##ध   │ 36 │
│  13 │ ##ा    │ 45 │
│  14 │ ##त   │ 34 │
│  15 │ ##ा    │ 45 │
└─────┴───────┴────┘
Encoding: जय हे जय हे 
        जय हे        
┏━━━━━┳━━━━━━━┳━━━━┓
┃ Pos ┃ Token ┃ ID ┃
┡━━━━━╇━━━━━━━╇━━━━┩
│   0 │ जय    │ 74 │
│   1 │ ##    │  0 │
│   2 │ ##ह   │  0 │
│   3 │ ##े    │ 50 │
│   4 │ ##    │  0 │
│   5 │ ##ज   │  0 │
│   6 │ ##य   │ 41 │
│   7 │ ##    │  0 │
│   8 │ ##ह   │  0 │
│   9 │ ##े    │ 50 │
│  10 │ ##    │  0 │
│  11 │ ##ज   │  0 │
│  12 │ ##य   │ 41 │
│  13 │ ##    │  0 │
│  14 │ ##ह   │  0 │
│  15 │ ##े    │ 50 │
└─────┴───────┴────┘
 Encoding: जय जय जय 
        जय हे        
┏━━━━━┳━━━━━━━┳━━━━┓
┃ Pos ┃ Token ┃ ID ┃
┡━━━━━╇━━━━━━━╇━━━━┩
│   0 │ जय    │ 74 │
│   1 │ ##    │  0 │
│   2 │ ##ज   │  0 │
│   3 │ ##य   │ 41 │
│   4 │ ##    │  0 │
│   5 │ ##ज   │  0 │
│   6 │ ##य   │ 41 │
│   7 │ ##    │  0 │
│   8 │ ##ज   │  0 │
│   9 │ ##य   │ 41 │
│  10 │ ##    │  0 │
│  11 │ ##ह   │  0 │
│  12 │ ##े    │ 50 │
└─────┴───────┴────┘
```

The tokenizer outputs a table containing:

* Token position
* Token text
* Token ID

---

# 5. Evaluation Metrics

## Token Count

Token count simply measures how many tokens the tokenizer produced for the input.

Script used:

```python
from pathlib import Path
from abctokz import Tokenizer

model = r"artifacts\task3_anthem_bpe"
eng_file = r"data\input_english_national_anthem.txt"
dev_file = r"data\input_devanagari_national_anthem.txt"

tok = Tokenizer.load(model)

def read_lines(p):
    return [x.strip() for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]

def token_count(lines):
    return sum(len(tok.encode(line)) for line in lines)

eng_lines = read_lines(eng_file)
dev_lines = read_lines(dev_file)

print("English token count:", token_count(eng_lines))
print("Devanagari token count:", token_count(dev_lines))
```

### Output

```
English token count: 317
Devanagari token count: 227
```

---

## Fertility

The main metric used in this experiment is **Fertility**.

### Definition

```
Fertility = Tokens / Words
```

Fertility measures **how many tokens are produced per word**.

* **Higher fertility → more token fragmentation**
* **Lower fertility → more efficient tokenization**

---

# 6. Results

| Script                  | Tokens | Words | Fertility |
| ----------------------- | ------ | ----- | --------- |
| English Transliteration | 317    | 55    | 5.763     |
| Devanagari              | 227    | 55    | 4.203     |

### Observation

The **English transliteration produces significantly more tokens** than the Devanagari version.

This means:

> The English transliteration is **less token-efficient**, because each word is split into more pieces.

---

# 7. Why Do the Token Counts Differ?

Even though both texts represent the **same anthem**, the token counts differ for several reasons.

### 1. Script Structure

The **Latin script** represents sounds using individual letters.

For example:

```
Adhinayaka
```

This becomes multiple character combinations that the tokenizer may not recognize, forcing it to split the word into many pieces.

In contrast, **Devanagari characters often represent richer phonetic units**, allowing the tokenizer to form slightly larger subword pieces.

---

### 2. Limited Training Data

Our tokenizer was trained on **only the anthem text**, which is an extremely small dataset.

Because of this:

* The tokenizer could not learn many useful subword merges
* Words are broken into smaller fragments

This increases the total number of tokens.

---

### 3. Vocabulary Size

The learned vocabulary size is **very small (87 tokens)**.

With such limited vocabulary coverage:

* Many words cannot be represented as larger subwords
* The tokenizer must fall back to smaller character-level pieces

---

### Conclusion

The difference is caused by a **combination of factors**:

* Script structure
* Limited training data
* Small vocabulary size

Together, these factors influence how efficiently the tokenizer can represent the text.

---

# 8. Bonus Experiment — GPT-4 Tokenizer (tiktoken)

To compare with a production-grade tokenizer, the same text was encoded using **GPT-4's tokenizer via the `tiktoken` library**.

### Installation

```
pip install tiktoken
```

---

## Script

```python
from pathlib import Path
import tiktoken

eng_file = r"data\input_english_national_anthem.txt"
dev_file = r"data\input_devanagari_national_anthem.txt"

enc = tiktoken.encoding_for_model("gpt-4")

def read_text(p):
    return Path(p).read_text(encoding="utf-8").strip()

def word_count(text):
    return len(text.split())

eng_text = read_text(eng_file)
dev_text = read_text(dev_file)

eng_tokens = enc.encode(eng_text)
dev_tokens = enc.encode(dev_text)

eng_words = word_count(eng_text)
dev_words = word_count(dev_text)

print("English tokens:", len(eng_tokens))
print("Devanagari tokens:", len(dev_tokens))
```

---

# 9. GPT-4 Tokenization Results

```
English Transliteration
tokens: 130
words: 55
fertility: 2.364

Devanagari
tokens: 276
words: 54
fertility: 5.111
```

---

# 10. Comparing the Tokenizers

| Tokenizer        | English Tokens | Devanagari Tokens |
| ---------------- | -------------- | ----------------- |
| Our BPE          | 317            | 227               |
| GPT-4 (tiktoken) | 130            | 276               |

---

# 11. What Does This Reveal?

The comparison highlights how **training scale and vocabulary size impact tokenization quality**.

### Our BPE Tokenizer

* Trained on **very small data**
* Small vocabulary
* Cannot learn strong subword patterns
* Words are heavily fragmented

### GPT-4 Tokenizer

* Trained on **massive multilingual datasets**
* Very large optimized vocabulary
* Already understands common patterns across many languages

As a result:

> GPT-4 can represent many words using **fewer tokens**, making it far more efficient.

---

# 12. Final Insight

This experiment demonstrates an important property of tokenization:

> **Token efficiency strongly depends on the tokenizer’s training data and vocabulary coverage.**

A tokenizer trained on limited text will split words into many small pieces, while a tokenizer trained on large multilingual corpora can encode the same text using far fewer tokens.


---







# Task 4 — How Does a Config Become a Tokenizer?


## 1) Where do default values come from?

Defaults come from **two places**:

### A) Preset functions 

Location:
- `src/abctokz/config/defaults.py`

Example preset used in this trace:
- `bpe_multilingual(vocab_size: int = 8000)`

This function constructs a `TokenizerConfig` with:
- a normalizer preset (sequence)
- a pre-tokenizer preset (sequence)
- a model config (`BPEConfig`)
- a trainer config (`BPETrainerConfig`)

Evidence command (prints the config object):

### B) Pydantic schema defaults 

Schema definitions live in:
- `src/abctokz/config/schemas.py`

Evidence command:

```powershell
python -c "from abctokz.config.schemas import BPEConfig,BPETrainerConfig; print(BPEConfig()); print(BPETrainerConfig())"
```

Paste output here:

```text
type='bpe' unk_token='<unk>' vocab_size=8000 continuation_prefix='##' end_of_word_suffix=''
type='bpe' vocab_size=8000 min_frequency=2 special_tokens=['<unk>'] limit_alphabet=None initial_alphabet=[] continuing_subword_prefix='##' end_of_word_suffix='' show_progress=True seed=42
```

Also defaults are pulled from constants present in:
- `src/abctokz/constants.py`

---

## 2) Where does validation happen (and what does it catch)? (Validation + evidence + failure modes)

Validation happens when the Pydantic objects are created.

Location:
- `src/abctokz/config/schemas.py`

This catches:
- typos / unknown keys in configs

### B) Numeric constraints (examples)
Examples in `schemas.py`:
- `vocab_size: int = Field(..., ge=1)`
- `min_frequency: int = Field(..., ge=1)`

This catches:
- invalid sizes like `vocab_size=0`

### C) Cross-field validation: model/trainer alignment
`TokenizerConfig` has a validator:
- `check_trainer_model_alignment()`

Location:
- `src/abctokz/config/schemas.py`

This catches:
- mismatched `model.type` vs `trainer.type` (e.g., BPE model + WordLevel trainer)

---

#### Failure mode #1: invalid vocab_size
What I tried:
- `BPEConfig(vocab_size=0)`

Expected:
- a Pydantic validation error because `vocab_size` has `ge=1`

Paste error here:

```text
vocab_size
  Input should be greater than or equal to 1 [type=greater_than_equal, input_value=0, input_type=int]
```

Was the message helpful?
- Yes it is helpful, as we get in error the field, type as well as the message

#### Failure mode #2: model/trainer mismatch
What I tried:
- `TokenizerConfig(model=BPEConfig(...), trainer=WordLevelTrainerConfig(...))`

Expected:
- a validation error from `TokenizerConfig.check_trainer_model_alignment()`

Paste error here:

```text
Value error, Model type 'bpe' and trainer type 'wordlevel' must match. [type=value_error, input_value={'model': BPEConfig(type=...progress=True, seed=42)}, input_type=dict]
```

Was the message helpful?
- Yes it is helpful, as we get in error the field, type as well as the message

---

## 3) Construction trace: config → normalizer → pre-tokenizer → model → trained tokenizer

The construction path splits into two phases:

### Phase 1: config → constructed pipeline objects

1) Build normalizer from config
- factory: `build_normalizer(...)`
- location: `src/abctokz/normalizers/__init__.py`

2) Build pre-tokenizer from config
- factory: `build_pretokenizer(...)`
- location: `src/abctokz/pretokenizers/__init__.py`

3) Build tokenizer shell
- `Tokenizer.from_config(config)`
- location: `src/abctokz/tokenizer.py`

- Uptil this point the model is a `_PlaceholderModel()` and `encode()` will fail until trained

### Phase 2: training → real model instance

4) Train in-place
- `Tokenizer.train(corpus_paths, config)`
- location: `src/abctokz/tokenizer.py`

Inside `train()`:
- trainer is constructed via `build_trainer(config.trainer)` in `src/abctokz/trainers/__init__.py`
- then `trainer.train(_corpus_iter())` returns a trained model object
  - e.g. for BPE: `BPETrainer.train(...)` returns `BPEModel`
  - location: `src/abctokz/trainers/bpe_trainer.py`

### Evidence for this question [script + output]

Model construction script:

```python
from __future__ import annotations

import sys
from pathlib import Path


def _force_utf8_stdout() -> None:
    # Helps on Windows terminals that default to a legacy codepage.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def _section(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{title}\n{bar}")


def _show_exception(label: str, exc: Exception) -> None:
    print(f"\n[{label}] {type(exc).__name__}:\n{exc}\n")


def main() -> int:
    _force_utf8_stdout()

    _section("Task 4 trace: config → normalizer → pre-tokenizer → model → trained tokenizer")

    from abctokz.config.defaults import bpe_multilingual
    from abctokz.normalizers import build_normalizer
    from abctokz.pretokenizers import build_pretokenizer
    from abctokz.trainers import build_trainer
    from abctokz.tokenizer import Tokenizer

    _section("1) Config")
    cfg_small = bpe_multilingual(vocab_size=200)
    print("Preset used: bpe_multilingual(vocab_size=200)")
    print(cfg_small)

    _section("2) Construction (config → normalizer/pretokenizer/trainer)")

    normalizer = build_normalizer(cfg_small.normalizer) if cfg_small.normalizer else None
    pretokenizer = build_pretokenizer(cfg_small.pretokenizer) if cfg_small.pretokenizer else None
    trainer = build_trainer(cfg_small.trainer) if cfg_small.trainer else None

    print("normalizer:", type(normalizer))
    print("pretokenizer:", type(pretokenizer))
    print("trainer:", type(trainer))

    _section("3) Tokenizer shell (from_config)")
    tok = Tokenizer.from_config(cfg_small)
    print("Tokenizer (before training):", tok)

    print("\nAttempting encode() before training (expected to fail):")
    try:
        _ = tok.encode("hello")
    except Exception as exc:
        _show_exception("encode before training", exc)

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = data_dir / "task4_corpus.txt"
    corpus_path.write_text(
        "hello world\nनमस्ते दुनिया\nhello नमस्ते world\n",
        encoding="utf-8",
    )

    _section("4) Training (trainer.train() returns a Model)")
    tok.train([str(corpus_path)], cfg_small)
    print("Tokenizer (after training):", tok)
    print("vocab_size:", tok.get_vocab_size())

    sample = "नमस्ते world"
    enc = tok.encode(sample)
    print("\nSample:", sample)
    print("tokens:", enc.tokens)
    print("ids:", enc.ids)
    print("decode:", tok.decode(enc.ids))

    _section("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

```

Output:

```text
Task 4 trace: config → normalizer → pre-tokenizer → model → trained tokenizer
=============================================================================

1) Config
=========
Preset used: bpe_multilingual(vocab_size=200)
schema_version='1' normalizer=SequenceNormalizerConfig(type='sequence', normalizers=[DevanagariNormalizerConfig(type='devanagari', nfc_first=True, strip_zero_width=False), WhitespaceNormalizerConfig(type='whitespace', strip=True, collapse=True)]) pretokenizer=SequencePreTokenizerConfig(type='sequence', pretokenizers=[DevanagariAwarePreTokenizerConfig(type='devanagari_aware', split_on_whitespace=True, split_on_script_boundary=True)]) model=BPEConfig(type='bpe', unk_token='<unk>', vocab_size=200, continuation_prefix='##', end_of_word_suffix='') trainer=BPETrainerConfig(type='bpe', vocab_size=200, min_frequency=2, special_tokens=['<unk>'], limit_alphabet=None, initial_alphabet=[], continuing_subword_prefix='##', end_of_word_suffix='', show_progress=True, seed=42) add_bos=False add_eos=False bos_token='<s>' eos_token='</s>' pad_token='<pad>'

2) Construction (config → normalizer/pretokenizer/trainer)
==========================================================
normalizer: <class 'abctokz.normalizers.sequence.SequenceNormalizer'>
pretokenizer: <class 'abctokz.pretokenizers.sequence.SequencePreTokenizer'>
trainer: <class 'abctokz.trainers.bpe_trainer.BPETrainer'>

3) Tokenizer shell (from_config)
================================
Tokenizer (before training): Tokenizer(model='unknown', vocab_size=0)

Attempting encode() before training (expected to fail):

[encode before training] RuntimeError:
Tokenizer has not been trained yet.


4) Training (trainer.train() returns a Model)
=============================================
Tokenizer (after training): Tokenizer(model='bpe', vocab_size=27)
vocab_size: 27

Sample: नमस्ते world
tokens: ['न', '##मस', '##्', '##ते', 'w', '##or', '##ld']
ids: [25, 17, 20, 13, 23, 10, 7]
decode: नमस्ते world

Done
====
```

---


---





# Task 5 — Is It Truly Deterministic?

## Experiment Setup

To verify the claim that the tokenizer training process is deterministic, the same tokenizer was trained **twice** using:

* **Model:** BPE
* **Vocabulary Size:** 200
* **Corpus:** `data/corpus.txt`
* **Configuration:** identical for both runs

Two independent training runs were executed:

```
artifacts/run1
artifacts/run2
```

Both models were then used to encode the same test input.

---

# Experiment 

Script to compare encoded output of both models
Here in this script , they validate if the encoded ouputs are identical and also campare the file structure of the tokenizer (e.g. vocab.json, merges.json, etc.)
```bash
@'
from pathlib import Path
from abctokz import Tokenizer
import hashlib

run1 = r"artifacts\run1"
run2 = r"artifacts\run2"
test_file = r"data\test.txt"

tok1 = Tokenizer.load(run1)
tok2 = Tokenizer.load(run2)

text = Path(test_file).read_text(encoding="utf-8")

enc1 = tok1.encode(text)
enc2 = tok2.encode(text)

print("Encoded tokens identical:", enc1 == enc2)

print("\nRun1 tokens:", enc1)
print("\nRun2 tokens:", enc2)

def file_hash(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()

files = ["vocab.json", "merges.txt"]

print("\nFile Comparisons")
for f in files:
    p1 = Path(run1) / f
    p2 = Path(run2) / f

    if p1.exists() and p2.exists():
        h1 = file_hash(p1)
        h2 = file_hash(p2)
        print(f"{f}: identical =", h1 == h2)
    else:
        print(f"{f}: file missing")
'@ | python -
```

Output

```
Encoded tokens identical: True

Run1 tokens: Encoding(n_tokens=99, tokens=['J','##a','##n','##a','## ', ...])

Run2 tokens: Encoding(n_tokens=99, tokens=['J','##a','##n','##a','## ', ...])

File Comparisons
vocab.json: identical = True
merges.txt: identical = True
```

Both runs also produced the same artifact files:

```
config.json
manifest.json
merges.txt
special_tokens.json
vocab.json
```

---

# What Parts Are Deterministic?

The experiment confirms that several components of the tokenizer pipeline are **fully deterministic**.

### 1. Vocabulary Generation

The `vocab.json` files were identical across runs.

This means that:

* Token frequencies were computed consistently
* Subword merges were learned in the same order
* Token IDs were assigned deterministically

---

### 2. Merge Rule Learning (BPE)

The `merges.txt` files were identical.

This indicates that the **BPE merge algorithm consistently selected the same most frequent pair at every step**, leading to the same merge sequence.

---

### 3. Encoding Output

Encoding the same input text produced:

* identical token sequences
* identical token IDs
* identical token counts

This confirms that **inference is deterministic when the model artifacts are identical**.

---

# What Parts Are Not Strictly Deterministic?

Even though the tokenizer itself behaves deterministically, some aspects of the process may vary slightly.

### Benchmark Timing

Training time and encoding speed may differ slightly between runs due to:

* CPU scheduling
* background processes
* system load

This does **not affect tokenizer correctness**, so it is acceptable.

---

# Remaining Risks — When Could Results Differ?

Even deterministic algorithms can produce different outputs under certain conditions.

### 1. Corpus Order Changes

If the corpus lines are shuffled, the frequency counts might be processed in a different order during tie situations, which could alter merge decisions.

---

### 2. Frequency Tie-Breaking

If two character pairs have **exactly the same frequency**, the algorithm must choose one first.

If the implementation does not enforce a deterministic tie-breaking rule, merge order could change.

---

### 3. Different Software Versions

Using different versions of:

* the tokenizer library
* Python
* dependencies

could potentially affect behavior.

---

### 4. Parallel Processing

If training were parallelized in the future, race conditions could introduce non-deterministic ordering unless explicitly controlled.

---

# Conclusion

The experiment demonstrates that the tokenizer training pipeline is **deterministic under controlled conditions**.

Training the same tokenizer twice with the same corpus and configuration produced:

* identical vocabularies
* identical merge rules
* identical encoded outputs

The only non-deterministic aspects are **external factors such as runtime performance**, which do not affect the correctness of the tokenizer.






# Task 6 — `<unk>` cases

## What I did

1) Trained 3 **English-only** tokenizers (so Devanagari + emoji are out-of-vocab).
2) Ran a few inputs and checked where ID `0` which is also (UNK) shows up.

### Quick script + output (small table)

Script I ran:

```powershell
python -X utf8 task-scripts\task_5_quick_run.py
```

`task-scripts/task_5_quick_run.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path


def _force_utf8_stdout() -> None:
	if hasattr(sys.stdout, "reconfigure"):
		sys.stdout.reconfigure(encoding="utf-8")
	if hasattr(sys.stderr, "reconfigure"):
		sys.stderr.reconfigure(encoding="utf-8")


def _ensure_src_on_path() -> None:
	"""Allow running from repo root without needing an editable install."""
	try:
		import abctokz  # noqa: F401
		return
	except Exception:
		repo_root = Path(__file__).resolve().parents[1]
		src_dir = repo_root / "src"
		if src_dir.exists():
			sys.path.insert(0, str(src_dir))


def _short_list(items: list[object], max_items: int = 8) -> str:
	if len(items) <= max_items:
		return str(items)
	head = ", ".join(repr(x) for x in items[:max_items])
	return f"[{head}, ...] (len={len(items)})"


def _unk_count(ids: list[int], unk_id: int) -> int:
	return sum(1 for i in ids if i == unk_id)


def main() -> int:
	_force_utf8_stdout()
	_ensure_src_on_path()

	from abctokz.constants import UNK_ID, UNK_TOKEN
	from abctokz.tokenizer import Tokenizer

	models = {
		"wordlevel": Path("artifacts") / "task5_eng_wordlevel",
		"bpe": Path("artifacts") / "task5_eng_bpe",
		"unigram": Path("artifacts") / "task5_eng_unigram",
	}

	tokenizers = {name: Tokenizer.load(str(path)) for name, path in models.items()}

	unk_id = UNK_ID
	for tok in tokenizers.values():
		got = tok.token_to_id(UNK_TOKEN)
		if got is not None:
			unk_id = got
			break

	tests: list[tuple[str, str]] = [
		("rare_en", "antidisestablishmentarianism"),
		("devanagari", "नमस्ते दुनिया"),
		("emoji", "🙂"),
		("mixed", "hello🙂world"),
	]

	print(f"unk_id={unk_id}\n")
	header = f"{'case':<10} {'model':<9} {'unk#':<4} tokens (short)"
	print(header)
	print("-" * len(header))
	for case, text in tests:
		for model_name, tok in tokenizers.items():
			enc = tok.encode(text)
			unk_n = _unk_count(enc.ids, unk_id)
			tokens_short = _short_list(enc.tokens, max_items=10)
			print(f"{case:<10} {model_name:<9} {unk_n:<4} {tokens_short}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
```

Output:

```text
unk_id=0

case       model     unk# tokens (short)
----------------------------------------
rare_en    wordlevel 1    ['<unk>']
rare_en    bpe       0    ['a', '##n', '##t', '##i', '##d', '##i', '##s', '##e', '##s', '##t', ...] (len=28)
rare_en    unigram   28   ['<unk>', '<unk>', '<unk>', '<unk>', '<unk>', '<unk>', '<unk>', '<unk>', '<unk>', '<unk>', ...] (len=28)
devanagari wordlevel 1    ['<unk>']
devanagari bpe       13   ['न', '##म', '##स', '##्', '##त', '##े', '## ', '##द', '##ु', '##न', ...] (len=13)
devanagari unigram   13   ['<unk>', '<unk>', '<unk>', '<unk>', '<unk>', '<unk>', '<unk>', '<unk>', '<unk>', '<unk>', ...] (len=13)
emoji      wordlevel 1    ['<unk>']
emoji      bpe       1    ['🙂']
emoji      unigram   1    ['<unk>']
mixed      wordlevel 1    ['<unk>']
mixed      bpe       1    ['h', '##e', '##l', '##l', '##o', '##🙂', '##w', '##o', '##r', '##l', ...] (len=11)
mixed      unigram   1    ['hello', '<unk>', 'world']
```


## When `<unk>` happened (and why)

### 1) WordLevel: whole-word OOV (model limitation)

Input: `antidisestablishmentarianism`

```text
wordlevel tokens: ['<unk>']
wordlevel ids   : [0]
```

Reason: WordLevel does exact lookup. If the whole token is not in vocab, it becomes `<unk>`.

### 2) Unseen characters (training corpus coverage)

Input: `नमस्ते दुनिया` (Devanagari) and `🙂` (emoji)

```text
bpe (Devanagari) ids   : [0, 0, 0, ...]
unigram (emoji) tokens : ['<unk>']
```

Reason: the English-only corpus never contained these characters, so BPE/Unigram can’t represent them and fall back to UNK.

### 3) Token boundary situation (mixed text)

Input: `hello🙂world`

```text
wordlevel tokens: ['<unk>']
unigram  tokens : ['hello', '<unk>', 'world']
```

Reason: without spaces, WordLevel sees one big unknown token. Unigram keeps the known parts and only marks the emoji as unknown.

---

## Which model is most graceful / most fragile?

- Most fragile: **WordLevel** (one unknown character can make the whole word become `<unk>`).
- Most graceful (for rare English words): **BPE** (it segmented a long unseen English word into known pieces and decoded back).

---

## One way to reduce UNK without retraining

Preprocess the input: [In case of emojis]
- insert spaces around symbols/emoji (`hello🙂world` → `hello 🙂 world`)
- or replace emoji with a plain word (`🙂` → `smiley`) before encoding




# Task 7 — Does Encode → Decode Get You Back to Start?

A tokenizer should ideally be **lossless**. This means that if we:

```
text → encode → decode
```

we should get **exactly the same text back**.

To verify this, I tested the tokenizer on several multilingual examples including:

* English text
* Devanagari (Hindi)
* Words with accented characters
* Unicode edge cases

**File:** `roundtrip_test.txt`

```
Hello world tokenizer test
Jana Gana Mana Adhinayaka Jaya He
जन गण मन अधिनायक जय हे
नमस्ते दुनिया
Cafe
Café
Café
```

---

# Test Method

Each sentence was passed through the tokenizer using the following pipeline:

```
original_text → tokenizer.encode() → tokenizer.decode()
```

Then the decoded output was compared with the original string.

---

# Round-Trip 
script to extract text from input then pass it as text for encoding and then perform decoding on the encoding ids
```bash
@'
>> from pathlib import Path
>> from abctokz import Tokenizer
>>
>> model = r"artifacts\task7_tokenizer"
>> test_file = r"data\roundtrip_test.txt"
>>
>> tok = Tokenizer.load(model)
>>
>> lines = Path(test_file).read_text(encoding="utf-8").splitlines()
>>
>> print("\nROUND TRIP TEST\n")
>>
>> success = 0
>>
>> for text in lines:
>>     if not text.strip():
>>         continue
>>
>>     enc = tok.encode(text)
>>     dec = tok.decode(enc.ids)
>> 
>>     same = text == dec
>>     
>>     print("Original :", repr(text))
>>     print("Decoded  :", repr(dec))
>>     print("Match    :", same)
>>     print("Tokens   :", enc.tokens)
>>     print("-"*50)
>> 
>>     if same:
>>         success += 1
>> 
>> print("\nTotal:", len(lines))
>> print("Exact matches:", success)
>> print("Success rate:", success/len(lines))
>> '@ | python -
```

Results
| Original Text                     | Decoded Output   | Match |
| --------------------------------- | ---------------- | ----- |
| Hello world tokenizer test        | *(empty string)* | No    |
| Jana Gana Mana Adhinayaka Jaya He | aaaaaaaaaaa      | No    |
| जन गण मन अधिनायक जय हे           | *(empty string)* | No    |
| नमस्ते दुनिया                         | *(empty string)* | No    |
| Cafe                              | Caf               | No    |
| Café                              | Café              | Yes   |
| Café                              | Caf               | No    |

### Summary

```
Total tests: 7
Exact matches: 1
Round trip success rate: 0.142857
```

Only **1 out of 7 cases** produced an exact round-trip.

---

# Case 1 — Exact Round Trip (Lossless)

Example:

```
Original: Café
Decoded : Café
Match   : True
```

### Why this works

The tokenizer contains a **single token for the composed character `é`**, so the encoding and decoding process preserves the word exactly.

```
Café → [C, ##a, ##f, ##é] → Café
```

Since the character is stored as a **single Unicode codepoint**, decoding reconstructs the original string correctly.

---

# Case 2 — Lossy Round Trip

Example:

```
Original: Cafe
Decoded : Caf
```

### What changed?

The final character **`e` disappeared** during decoding.

Tokens:

```
[C, ##a, ##f, ##e]
```

During decoding, the tokenizer failed to correctly reconstruct the last token.

### Why this happens

This likely occurs because:

* the tokenizer uses **subword tokens (`##`)**
* the decoding logic may **incorrectly strip or merge suffix tokens**

This causes the reconstructed string to lose characters.

---

# Unicode Edge Case — NFC vs NFD

Another interesting example is:

```
Original: Café
Decoded : Caf
```

At first glance, `Café` and `Café` look identical. However, they are **different Unicode representations**.

### NFC (composed form)

```
Café
```

Character sequence:

```
C + a + f + é
```

### NFD (decomposed form)

```
Café
```

Character sequence:

```
C + a + f + e + ◌́
```

Here the accent is a **separate combining character**.

The tokenizer splits it as:

```
[C, ##a, ##f, ##e, ##́]
```

During decoding, the combining accent is not reconstructed correctly, resulting in:

```
Caf
```

---

# Is the Lossy Behavior a Bug?

This depends on the intended design.

Possible explanations:

### 1. Implementation Bug

If the tokenizer is supposed to be **fully reversible**, then losing characters during decoding is a **bug in the decode logic**.

### 2. Acceptable Trade-off

Some tokenizers sacrifice perfect reconstruction for **simpler token rules**, especially when using subword prefixes like `##`.

### 3. Unicode Normalization Issues

Unicode normalization differences (NFC vs NFD) can produce visually identical text that is **not byte-identical**, which may cause decoding differences.

---

# What `round_trip_success_rate` Measures

The metric `round_trip_success_rate` checks whether:

```
decoded_text == original_text
```

If they match exactly, the test counts as successful.

So the metric measures:

* Exact string equality
* Whether encode → decode preserves the text perfectly

---

# What the Metric Does NOT Measure

However, this metric does **not detect several important cases**.

### 1. Visual Equality

Two strings can look identical but still fail the equality test.

Example:

```
Café (NFC)
Café (NFD)
```

They look the same but have **different Unicode codepoints**.

---

### 2. Semantic Equivalence

The metric does not check if the decoded text **still has the same meaning**.

Example:

```
Original: Cafe
Decoded : Caf
```

The metric simply flags it as failure but does not explain **why**.

---

### 3. Partial Reconstruction Errors

If a tokenizer drops or modifies characters, the metric only reports **failure**, not which tokens caused it.

---

# Key Insight

A tokenizer may appear to work correctly in most cases, but **Unicode normalization and subword decoding rules can introduce subtle reconstruction errors**.

This experiment shows that:

* exact round-trip behavior is **not guaranteed**
* Unicode representation plays an important role
* round-trip evaluation should consider **both byte equality and Unicode normalization**





# Task 8 — What Does the Normalizer Actually Do?

## Setup — which normalizer does the library actually use for Devanagari?

From [`src/abctokz/config/defaults.py`](../src/abctokz/config/defaults.py), the `devanagari_safe_normalizer()` preset used by all multilingual configs is:

```python
SequenceNormalizerConfig(normalizers=[
    DevanagariNormalizerConfig(nfc_first=True, strip_zero_width=False),
    WhitespaceNormalizerConfig(strip=True, collapse=True),
])
```

So the pipeline is:  
**raw text → NFC → exotic-whitespace normalization → whitespace strip/collapse → pre-tokenizer**

No NFKC is ever applied to Devanagari text. This is intentional (see Section 2 below).

---

## Q1 — Raw input vs normalized output: are they identical?

Both phrases were run through the full pipeline. Output:

```
--- SINDHI ---
  after NFC        : 'आयो लाल, सभई चायो, झूलेलाल!'
  after DevanagariN: 'आयो लाल, सभई चायो, झूलेलाल!'
  after Whitespace : 'आयो लाल, सभई चायो, झूलेलाल!'
  raw == final     : True

--- MARATHI ---
  after NFC        : 'गणपती बप्पा मोरया, पुढच्या वर्षी लवकर या!'
  after DevanagariN: 'गणपती बप्पा मोरया, पुढच्या वर्षी लवकर या!'
  after Whitespace : 'गणपती बप्पा मोरया, पुढच्या वर्षी लवकर या!'
  raw == final     : True
```

**Both phrases are already in NFC form and have no exotic whitespace — so the normalized output is byte-for-byte identical to the raw input.** The normalizer is a no-op here, which is actually correct behavior: well-formed Devanagari Unicode text that is already NFC needs no transformation.

The normalizer would visibly change the input only if:
- The input were in NFD form (decomposed combining marks stored separately), or
- There were exotic Unicode spaces (e.g. `U+00A0 NO-BREAK SPACE`), or
- `strip_zero_width=True` and ZWJ/ZWNJ characters were present.

---

## Q2 — NFC vs NFKC: which does this library use and why?

| Form | What it does | Safe for Devanagari? |
|---|---|---|
| **NFC** | Canonical Decomposition then Canonical Composition. Reorders and recomposes combining marks without changing character semantics. | **Yes** — matras, halant, anusvara stay semantically intact. |
| **NFKC** | Compatibility Decomposition (lossy) then Canonical Composition. Folds "compatibility equivalents" into base forms. | **No** — can collapse or change Devanagari combining marks in ways that alter pronunciation. |

**This library uses NFC for Devanagari.** The code comment in [`src/abctokz/normalizers/devanagari.py`](../src/abctokz/normalizers/devanagari.py) explicitly states:

> *We apply NFC (not NFKC) because NFKC can collapse Devanagari combining marks in ways that change the visual and phonetic form of the character.*

Verified experimentally — for both phrases, NFC == NFKC happened to be true here because the inputs contain no compatibility characters. But the danger is real for other inputs, for example fullwidth digits (U+FF10–U+FF19) or certain Devanagari extended characters would be altered irreversibly by NFKC.

The `NfkcNormalizer` class (used for English) carries a `.. warning::` in its docstring:
> *NFKC can be lossy for some Devanagari text. For Devanagari input, prefer `DevanagariNormalizer` which uses NFC instead.*

---

## Q3 — What happens to commas, exclamation mark, and spaces after pre-tokenization?

The `DevanagariAwarePreTokenizer` splits **only on whitespace** (first pass), then optionally on script boundaries (second pass). Punctuation is **not treated as a split signal** — it gets attached to the adjacent whitespace-delimited word.

Verified output from `python -c "..."`:

### Sindhi phrase: `'आयो लाल, सभई चायो, झूलेलाल!'`

```
pre-tokens : ['आयो', 'लाल,', 'सभई', 'चायो,', 'झूलेलाल!']
  [0] 'आयो'       — pure Devanagari word
  [1] 'लाल,'      — Devanagari word WITH comma glued on (U+002C Po COMMA)
  [2] 'सभई'       — pure Devanagari word
  [3] 'चायो,'     — Devanagari word WITH comma glued on
  [4] 'झूलेलाल!'  — Devanagari word WITH exclamation mark glued on (U+0021 Po)
```

### Marathi phrase: `'गणपती बप्पा मोरया, पुढच्या वर्षी लवकर या!'`

```
pre-tokens : ['गणपती', 'बप्पा', 'मोरया,', 'पुढच्या', 'वर्षी', 'लवकर', 'या!']
  [0] 'गणपती'    — pure
  [1] 'बप्पा'    — contains VIRAMA (U+094D) — conjunct preserved intact
  [2] 'मोरया,'   — word WITH comma
  [3] 'पुढच्या'  — contains VIRAMA (U+094D) in च्य conjunct — preserved intact
  [4] 'वर्षी'    — contains VIRAMA (U+094D) in र्ष conjunct — preserved intact
  [5] 'लवकर'     — pure
  [6] 'या!'      — word WITH exclamation mark
```

**Key findings:**
- **Spaces** (U+0020 `Zs` category): consumed as split boundaries, not kept as tokens.
- **Commas and `!`** (U+002C, U+0021, both `Po` — Other Punctuation): **not split on** — the `DevanagariAwarePreTokenizer` only splits on whitespace and script-change boundaries, never on punctuation. They ride along with the preceding/trailing Devanagari word.
- **Implication**: tokens `'लाल,'`, `'चायो,'`, `'मोरया,'`, `'झूलेलाल!'`, `'या!'` are pre-tokens that include punctuation. When the BPE/Unigram model tokenizes these pre-tokens, the comma and `!` become subword pieces alongside letter pieces. If the vocabulary doesn't contain `'लाल,'` as a unit, BPE will fall back to character-level pieces including `','`.

This is different from many Western tokenizers (e.g. Punkt, HF's `ByteLevelBPETokenizer`) which always isolate punctuation. There is no `PunctuationPreTokenizer` in the multilingual config.

---

## Q4 — Why does this matter specifically for Hindi, Marathi, and Sindhi?

### Virama and conjunct consonants

In Devanagari, a **Virama** (U+094D `्`) is a combining diacritic that suppresses the inherent vowel of a consonant, creating **conjunct consonants** (two consonants written as a single fused glyph). Examples verified in the Marathi phrase:

```
'बप्पा' chars:
  ब U+092C  DEVANAGARI LETTER BA
  प U+092A  DEVANAGARI LETTER PA
  ् U+094D  DEVANAGARI SIGN VIRAMA   ← suppresses PA's vowel
  प U+092A  DEVANAGARI LETTER PA
  ा U+093E  DEVANAGARI VOWEL SIGN AA

'पुढच्या' chars:
  ...
  च U+091A  DEVANAGARI LETTER CA
  ् U+094D  DEVANAGARI SIGN VIRAMA   ← forms च्य conjunct
  य U+092F  DEVANAGARI LETTER YA
  ा U+093E  DEVANAGARI VOWEL SIGN AA
```

The `grapheme_clusters()` function in [`src/abctokz/utils/unicode.py`](../src/abctokz/utils/unicode.py) keeps combining marks (category `Mn`, `Mc`) attached to their base characters. This ensures that `्` is never split away from its base consonant during pre-tokenization.

### ZWJ and ZWNJ — why `strip_zero_width=False` is the safe default

```
ZWJ  demo: raw='र्\u200dक'  preserved='र्\u200dक'  stripped='र्क'
ZWNJ demo: raw='र्\u200cक'  preserved='र्\u200cक'  stripped='र्क'
```

- **ZWJ** (U+200D): when placed between a consonant, VIRAMA, and the next consonant, it **forces a half-form** rendering rather than a fully fused conjunct. Used in Hindi and Marathi to write `र्` (half-ra) explicitly.
- **ZWNJ** (U+200C): forces the **explicit halant** (visible VIRAMA) rendering instead of conjunct formation. Critical in Sindhi, where some conjunct vs. non-conjunct distinctions are phonemically meaningful.

The library defaults to `strip_zero_width=False` because removing ZWJ/ZWNJ changes the **phonetic identity** of the word. `'र्‍क'` (with ZWJ) and `'र्क'` (without) look different on screen and may have different phonetics. Stripping them silently would be lossy.

For Sindhi specifically, ZWJ/ZWNJ are widely used in text from Pakistan and Indian Sindhi sources to correctly render Devanagari Sindhi, which has more conjunct rules than standard Hindi. Stripping them would cause tokenizer to produce different token sequences for what appears visually distinct to a native reader.

### Summary table

| Script feature | Unicode codepoint | What happens if stripped / mishandled |
|---|---|---|
| Matra (vowel sign) | U+093E–U+094C (Mc/Mn) | Word loses its vowel — phonetically wrong |
| Virama (halant) | U+094D | Conjuncts break apart — two consonants become unrelated |
| ZWJ | U+200D | Half-forms collapse to full conjunct — changes visual/phonetic form |
| ZWNJ | U+200C | Implicit conjunct forms — changes rendering and phonetics |
| Anusvara | U+0902 | Nasalization lost — changes word meaning |

The NFC choice + `strip_zero_width=False` default in `DevanagariNormalizer` is specifically designed to avoid all of these hazards. The comment in the source code (`devanagari.py` lines 5–18) explains this rationale directly.





# Task 9 — Measuring Phrase Difficulty

## Setup

### The two phrases (carried forward from Task 8)

| Label | Phrase | Language | Whitespace words |
|---|---|---|---|
| SINDHI | `आयो लाल, सभई चायो, झूलेलाल!` | Sindhi Devanagari | 5 |
| MARATHI | `गणपती बप्पा मोरया, पुढच्या वर्षी लवकर या!` | Marathi Devanagari | 7 |

Reference word count (for the fertility denominator) is computed with Python's `.split()` — whitespace-delimited tokens.  Punctuation (`,` and `!`) is **not** a split boundary, so it remains attached to the preceding word, exactly as Task 8 showed.

### Training corpus

A Devanagari-rich corpus of **75 unique sentences × 50 repetitions = 3 750 lines** was constructed in-process. It covers:
- **Hindi** (30 sentences): everyday phrases, geography, proverbs
- **Marathi** (20 sentences): includes the exact Ganpati phrase and related vocabulary
- **Sindhi Devanagari** (15 sentences): includes आयो, लाल, सभई, चायो, झूलेलाल and related words
- **English + mixed** (10 sentences)

The corpus was written to a temp file and passed to `Tokenizer.train()` via the standard API.  No save/load was performed — the tokenizer was used directly after training to avoid the known pipeline-gap bug (see Task 2).

### Fertility definition (from `src/abctokz/eval/metrics.py`)

```python
def fertility(encodings, reference_word_counts):
    total_tokens = sum(len(e) for e in encodings)
    total_ref    = sum(reference_word_counts)
    return total_tokens / total_ref
```

Applied per-phrase: `fertility = n_tokens / n_whitespace_words`.

---

## Results

### Training runs: actual vocab sizes produced

| Requested | BPE actual | Unigram actual |
|---|---|---|
| 100 | **125** | 100 |
| 400 | 400 | 400 |
| 800 | **710** | 800 |

BPE-100 inflated to 125 because the special-token set plus the Devanagari+Latin alphabet forces a minimum.  BPE-800 capped at 710 because the corpus exhausted available merge pairs before reaching 800.

---

### Fertility table

```
                        BPE-100  BPE-400  BPE-800  |   UNI-100  UNI-400  UNI-800
  -------------------- -------- -------- --------  |  -------- -------- --------
  SINDHI                  4.600    3.400    3.400  |     2.200    1.600    1.600
  MARATHI                 5.000    3.429    3.286  |     4.429    1.286    1.286
```

### Token count table (raw numbers before dividing by word count)

```
                        BPE-100  BPE-400  BPE-800  |   UNI-100  UNI-400  UNI-800
  SINDHI (÷ 5 words)       23       17       17   |       11        8        8
  MARATHI (÷ 7 words)      35       24       23   |       31        9        9
```

---

### Fertility reduction from vocab=100 to vocab=800

| Phrase | Algorithm | 100 → 400 → 800 | Total reduction |
|---|---|---|---|
| SINDHI | BPE | 4.600 → 3.400 → 3.400 | **26.1%** |
| SINDHI | Unigram | 2.200 → 1.600 → 1.600 | **27.3%** |
| MARATHI | BPE | 5.000 → 3.429 → 3.286 | **34.3%** |
| MARATHI | Unigram | 4.429 → 1.286 → 1.286 | **71.0%** |

---

## Q1 — Fertility for each phrase at each configuration

### BPE — detailed token piece breakdowns

**BPE, vocab=100 (actual 125)**

```
SINDHI  | tokens=23 | fertility=4.600
  pieces: ['आ', '##य', '##ो', 'ल', '##ा', '##ल', '##,', 'स', '##भ', '##ई',
           'च', '##ा', '##य', '##ो', '##,', 'झ', '##ू', '##ल', '##े', '##ल',
           '##ा', '##ल', '##!']
  UNK tokens: 3 at positions [6, 14, 22]  (the three punctuation marks)

MARATHI | tokens=35 | fertility=5.000
  pieces: ['ग', '##ण', '##प', '##त', '##ी', 'ब', '##प', '##्', '##प', '##ा',
           'म', '##ो', '##र', '##य', '##ा', '##,', 'प', '##ु', '##ढ', '##च',
           '##्', '##य', '##ा', 'व', '##र', '##्', '##ष', '##ी', 'ल', '##व',
           '##क', '##र', 'य', '##ा', '##!']
  UNK tokens: 2 at positions [15, 34]  (comma and !)
```

At vocab=100 BPE is overwhelmingly character-level. Every Devanagari character is a separate piece. MARATHI needs 35 pieces for 7 words (5.000) vs SINDHI's 23 for 5 words (4.600).  The `##` prefix marks a suffix-of-word continuation piece.

**BPE, vocab=400**

```
SINDHI  | tokens=17 | fertility=3.400
  pieces: ['आ', '##यो', 'ल', '##ाल', '##,', 'सभ', '##ई', 'च', '##ा', '##यो',
           '##,', 'झ', '##ू', '##ले', '##ल', '##ाल', '##!']
  UNK: 3 positions [4, 10, 16]

MARATHI | tokens=24 | fertility=3.429
  pieces: ['गण', '##प', '##ती', 'ब', '##प्', '##पा', 'मो', '##र', '##या',
           '##,', 'प', '##उढ', '##च', '##्', '##या', 'व', '##र्', '##ष', '##ी',
           'ल', '##व', '##कर', 'या', '##!']
  UNK: 2 positions [9, 23]
```

At 400 BPE starts using 2-character bigrams.  SINDHI words like `लाल` are learned as root + `##ाल`.  MARATHI's virama-bearing subwords (`##प्`, `##र्`, `##च्`) prevent them from merging further.

**BPE, vocab=800 (actual 710)**

```
SINDHI  | tokens=17 | fertility=3.400   (unchanged from 400)
  pieces: ['आ', '##यो', 'ल', '##ाल', '##,', 'सभ', '##ई', 'च', '##ा', '##यो',
           '##,', 'झ', '##ू', '##ले', '##ल', '##ाल', '##!']
  UNK: 3 positions [4, 10, 16]

MARATHI | tokens=23 | fertility=3.286   (one merge gained vs 400)
  pieces: ['गण', '##प', '##ती', 'ब', '##प्', '##पा', 'मो', '##र', '##या',
           '##,', 'प', '##ुढ', '##च्', '##या', 'व', '##र्', '##ष', '##ी',
           'ल', '##व', '##कर', 'या', '##!']
  UNK: 2 positions [9, 22]
```

The only change from 400→800 in MARATHI is that `##च` + `##्` merged into `##च्` (saving 1 token). SINDHI gained nothing.

---

### Unigram — detailed token piece breakdowns

**Unigram, vocab=100**

```
SINDHI  | tokens=11 | fertility=2.200
  pieces: ['आयो', 'लाल', '<unk>', 'सभई', '<unk>', '<unk>', '<unk>', '<unk>',
           '<unk>', 'झूलेलाल', '<unk>']
  UNK tokens: 7 at positions [2, 4, 5, 6, 7, 8, 10]

MARATHI | tokens=31 | fertility=4.429
  pieces: ['ग', 'ण', 'प', 'त', 'ी', '<unk>', '<unk>', '<unk>', '<unk>', '<unk>',
           'मोरया', '<unk>', 'प', '<unk>', '<unk>', '<unk>', '<unk>', '<unk>',
           '<unk>', 'व', 'र', '्', 'ष', 'ी', 'ल', 'व', 'क', 'र', 'य', 'ा', '<unk>']
  UNK tokens: 13 at positions [5,6,7,8,9,11,13,14,15,16,17,18,30]
```

At vocab=100 Unigram the contrast is extreme. SINDHI benefits because its exact words (`आयो`, `सभई`, `झूलेलाल`) appear verbatim in the training corpus and make it into the 100-entry vocabulary. MARATHI's virama-bearing forms (`गणपती`, `बप्पा`, `पुढच्या`, `वर्षी`) did not — they fall back to character pieces, many of which are also OOV → cascading UNKs (13 total).

**Unigram, vocab=400**

```
SINDHI  | tokens=8 | fertility=1.600
  pieces: ['आयो', 'लाल', '<unk>', 'सभई', 'चायो', '<unk>', 'झूलेलाल', '<unk>']
  UNK tokens: 3 at positions [2, 5, 7]   ← only the 3 punctuation marks

MARATHI | tokens=9 | fertility=1.286
  pieces: ['गणपती', 'बप्पा', 'मोरया', '<unk>', 'पुढच्या', 'वर्षी', 'लवकर', 'या', '<unk>']
  UNK tokens: 2 at positions [3, 8]   ← only the 2 punctuation marks
```

Unigram finds whole-word pieces at 400. Every actual Devanagari word in both phrases is now a single token. The only remaining UNKs are `,` and `!` — the punctuation marks that were never in the training corpus.

**Unigram, vocab=800**

```
SINDHI  | tokens=8 | fertility=1.600   (identical to 400)
MARATHI | tokens=9 | fertility=1.286   (identical to 400)
```

No change at all from 400 to 800 for either phrase.

---

## Q2 — Which phrase is harder to tokenize efficiently, and why?

### BPE answer: MARATHI is consistently harder

At every BPE vocabulary size, MARATHI has equal or higher fertility than SINDHI:

| Vocab | SINDHI | MARATHI | Winner |
|---|---|---|---|
| BPE-100 | 4.600 | **5.000** | MARATHI harder |
| BPE-400 | 3.400 | **3.429** | MARATHI harder |
| BPE-800 | **3.400** | 3.286 | SINDHI marginally harder |

The structural reason is directly observable in the piece breakdowns.  MARATHI contains three virama-mediated conjunct consonants:

| Word | Conjunct | Unicode structure |
|---|---|---|
| `बप्पा` | `ब+प्+पा` | ब (BA) + प (PA) + ् (VIRAMA) + प (PA) + ा (AA) |
| `पुढच्या` | `च्+या` | च (CA) + ् (VIRAMA) + य (YA) + ा (AA) |
| `वर्षी` | `र्+षी` | र (RA) + ् (VIRAMA) + ष (SSA) + ी (II) |

The VIRAMA (U+094D) acts as a combining diacritic that fuses two consonants visually.  BPE tries to merge byte-adjacent pairs greedily, but a sequence like `प + ् + प` must remain a 3-character sequence because splitting after `्` changes the rendering of the conjunct.  The tokenizer learns `##प्` as a subword but cannot merge it further with `##पा` without sufficient corpus evidence — resulting in the 3-piece breakdown `ब + ##प् + ##पा` instead of the ideal 1-piece `बप्पा`.

SINDHI words like `आयो`, `चायो`, `सभई` carry simple vowel matras (ा, ो, ई) but no viramas.  These merge more readily: `च + ##ायो` or `आ + ##यो`.

### Unigram answer: it depends on vocabulary size

| Vocab | SINDHI | MARATHI | Observation |
|---|---|---|---|
| UNI-100 | **2.200** | 4.429 | SINDHI much easier |
| UNI-400 | 1.600 | **1.286** | MARATHI becomes easier |
| UNI-800 | 1.600 | **1.286** | Same |

This reversal reveals a key difference: **frequency in training data, not linguistic complexity, governs Unigram difficulty at small vocabulary sizes**.

At vocab=100, the Unigram model can only keep 100 pieces.  Short, high-frequency Sindhi words (`आयो`, `सभई`, `झूलेलाल`) that appeared directly in the training corpus win limited vocabulary slots.  MARATHI test words are longer, have viramas, and some appeared less frequently → they don't make the cut → character-level fallback with cascading UNKs.

At vocab=400, the model has enough capacity to learn whole-word pieces for all Devanagari content words in both phrases.  Now MARATHI's advantage shows: its 7-word phrase encodes as 9 tokens (7 words + 2 punctuation UNKs), giving fertility 9/7=1.286, while SINDHI's shorter 5-word phrase encodes as 8 tokens (5 words + 3 punctuation UNKs) giving 8/5=1.600.  SINDHI's fertility suffers from punctuation density — 3 attached punctuation marks across only 5 words.

### Summary conclusion

**MARATHI is harder to tokenize efficiently with BPE** due to its virama-mediated conjunct consonants preventing subword merges.  
**The picture inverts with Unigram at larger vocabulary** because Unigram's probabilistic model can select whole-word segmentations when they exist in the vocabulary, and MARATHI's longer phrase dilutes the punctuation overhead.  
The punctuation (`,` and `!`) represents a confounding factor in both cases: they are OOV in this experiment because the training corpus contained no punctuation — their UNK penalty disproportionately hurts SINDHI (shorter phrase, same number of punctuation marks relative to word count).

---

## Q3 — Does fertility change meaningfully with vocabulary size?

### BPE: yes at 100→400, no at 400→800

```
BPE SINDHI:   4.600 → 3.400 → 3.400   (drops 26.1%, then plateaus)
BPE MARATHI:  5.000 → 3.429 → 3.286   (drops 31.4%, then minor -0.143)
```

The drop from 100 to 400 is substantial (≈ 26–31%) because the model gains bigram and trigram subword units for common character sequences.  By 400 most high-frequency character n-grams in the Devanagari alphabet have been merged.  From 400 to 800, SINDHI gains nothing (the corpus provides no new merge pairs for SINDHI words beyond what 400 already learned).  MARATHI gains one small merge (`##च` + `##्` → `##च्`), saving 1 token, which reduces fertility from 3.429 to 3.286 (−0.143).

**What this tells us**: BPE fertility saturates well before the requested vocabulary is reached.  Doubling the vocabulary from 400 to 800 only improved MARATHI by 4.2% and SINDHI by 0%.  This is because the corpus is smaller than the vocabulary target — BPE ran out of pair merges at 710 actual entries.

### Unigram: enormous jump at 100→400, then complete plateau at 400→800

```
UNI SINDHI:   2.200 → 1.600 → 1.600   (drops 27.3%, then zero change)
UNI MARATHI:  4.429 → 1.286 → 1.286   (drops 71.0%, then zero change)
```

The 100→400 jump for MARATHI (4.429 → 1.286, a 71% reduction) is dramatic.  This single step means the difference between nearly character-level tokenization and whole-word tokenization.  Unigram's training algorithm selects whole words once they're frequent enough; 400 entries is enough to fit all 7 unique Marathi words from the phrase.

The 400→800 plateau (zero change for both phrases and both algorithms) means **the test phrases themselves have saturated** — there are no new subword merges or vocabulary entries that would help encode these specific phrases any more efficiently at vocab=400.

### What the plateau tells us about vocabulary sizing

A plateau means the test phrases have **benefited fully** from the available vocabulary.  No further fertility improvement will come from a larger vocabulary unless: (a) punctuation is added to the training corpus, (b) the tokenizer uses a `PunctuationPreTokenizer` to isolate punctuation as its own tokens, or (c) the model is trained on text with more varied character n-grams that create more merge opportunities.

In production, fertility gains from increasing vocabulary typically follow a **diminishing-returns curve** — big gains early, then a long tail.  These results show that even a modest 400-entry vocabulary is sufficient for efficient encoding of short Devanagari phrases if the corpus is representative.

---

## BPE vs Unigram: structural comparison

| Property | BPE | Unigram |
|---|---|---|
| Segmentation algorithm | Greedy left-to-right merges | Maximum-probability subword lattice |
| Fertility at vocab=400 (Sindhi) | 3.400 | **1.600** |
| Fertility at vocab=400 (Marathi) | 3.429 | **1.286** |
| Sensitivity to virama conjuncts | High (can't merge across virama boundary easily) | Lower (can select whole-word paths) |
| UNK behaviour | Returns surface character as piece, assigns UNK ID | Returns literal `<unk>` piece |

Unigram produces lower fertility at every comparable vocab size when the corpus is representative.  The trade-off is that Unigram requires the exact word to be in vocabulary — any unseen word falls to character-level or `<unk>`.  BPE degrades more gracefully: it always produces some subword decomposition even for never-seen words (as long as the characters are in the alphabet).

---

## Connection to Task 8

| Task 8 observation | Task 9 confirmation |
|---|---|
| Commas and `!` glue to adjacent Devanagari words (not split boundaries) | Confirmed via UNK tokens at punctuation positions in every config |
| Virama (U+094D) preserves conjunct consonants in pre-tokens | Confirmed: `##प्`, `##र्`, `##च्` appear as separate subwords in BPE because they can't merge cleanly across the virama |
| MARATHI has more virama-conjuncts than SINDHI | Confirmed: MARATHI always needs more BPE tokens per word |
| `DevanagariAwarePreTokenizer` splits only on whitespace + script boundary | Confirmed: the pre-tokens fed to BPE/Unigram are exactly the whitespace-split groups (with glued punctuation) |

---

## Reproduction

```bash
# Script is at task9_experiment.py in the repo root
python task9_experiment.py
```

Full terminal output:
```
SINDHI  whitespace_words: ['आयो', 'लाल,', 'सभई', 'चायो,', 'झूलेलाल!'] -> 5
MARATHI whitespace_words: ['गणपती', 'बप्पा', 'मोरया,', 'पुढच्या', 'वर्षी', 'लवकर', 'या!'] -> 7
Corpus: 3750 lines

========================================================================
MODEL TYPE: BPE
========================================================================
  -- vocab_size=100 --
     trained vocab_size: 125 (requested 100)
     SINDHI   | tokens= 23 | words=5 | fertility=4.600
               pieces: ['आ', '##य', '##ो', 'ल', '##ा', '##ल', '##,', 'स', '##भ',
 '##ई', 'च', '##ा', '##य', '##ो', '##,', 'झ', '##ू', '##ल', '##े', '##ल', '##ा',
 '##ल', '##!']
               UNK tokens: 3 at positions [6, 14, 22]
     MARATHI  | tokens= 35 | words=7 | fertility=5.000
               pieces: ['ग', '##ण', '##प', '##त', '##ी', 'ब', '##प', '##्', '##प',
 '##ा', 'म', '##ो', '##र', '##य', '##ा', '##,', 'प', '##ु', '##ढ', '##च', '##्',
 '##य', '##ा', 'व', '##र', '##्', '##ष', '##ी', 'ल', '##व', '##क', '##र', 'य',
 '##ा', '##!']
               UNK tokens: 2 at positions [15, 34]

  -- vocab_size=400 --
     trained vocab_size: 400 (requested 400)
     SINDHI   | tokens= 17 | words=5 | fertility=3.400
               pieces: ['आ', '##यो', 'ल', '##ाल', '##,', 'सभ', '##ई', 'च', '##ा',
 '##यो', '##,', 'झ', '##ू', '##ले', '##ल', '##ाल', '##!']
               UNK tokens: 3 at positions [4, 10, 16]
     MARATHI  | tokens= 24 | words=7 | fertility=3.429
               pieces: ['गण', '##प', '##ती', 'ब', '##प्', '##पा', 'मो', '##र',
 '##या', '##,', 'प', '##ुढ', '##च', '##्', '##या', 'व', '##र्', '##ष', '##ी',
 'ल', '##व', '##कर', 'या', '##!']
               UNK tokens: 2 at positions [9, 23]

  -- vocab_size=800 --
     trained vocab_size: 710 (requested 800)
     SINDHI   | tokens= 17 | words=5 | fertility=3.400  (unchanged)
               pieces: [same as 400]
               UNK tokens: 3 at positions [4, 10, 16]
     MARATHI  | tokens= 23 | words=7 | fertility=3.286
               pieces: ['गण', '##प', '##ती', 'ब', '##प्', '##पा', 'मो', '##र',
 '##या', '##,', 'प', '##ुढ', '##च्', '##या', 'व', '##र्', '##ष', '##ी', 'ल',
 '##व', '##कर', 'या', '##!']
               UNK tokens: 2 at positions [9, 22]

========================================================================
MODEL TYPE: UNIGRAM
========================================================================
  -- vocab_size=100 --
     trained vocab_size: 100 (requested 100)
     SINDHI   | tokens= 11 | words=5 | fertility=2.200
               pieces: ['आयो', 'लाल', '<unk>', 'सभई', '<unk>', '<unk>', '<unk>',
 '<unk>', '<unk>', 'झूलेलाल', '<unk>']
               UNK tokens: 7 at positions [2, 4, 5, 6, 7, 8, 10]
     MARATHI  | tokens= 31 | words=7 | fertility=4.429
               pieces: ['ग', 'ण', 'प', 'त', 'ी', '<unk>', '<unk>', '<unk>',
 '<unk>', '<unk>', 'मोरया', '<unk>', 'प', '<unk>', '<unk>', '<unk>', '<unk>',
 '<unk>', '<unk>', 'व', 'र', '्', 'ष', 'ी', 'ल', 'व', 'क', 'र', 'य', 'ा', '<unk>']
               UNK tokens: 13 at positions [5-9,11,13-18,30]

  -- vocab_size=400 --
     trained vocab_size: 400 (requested 400)
     SINDHI   | tokens=  8 | words=5 | fertility=1.600
               pieces: ['आयो', 'लाल', '<unk>', 'सभई', 'चायो', '<unk>',
 'झूलेलाल', '<unk>']
               UNK tokens: 3 at positions [2, 5, 7]
     MARATHI  | tokens=  9 | words=7 | fertility=1.286
               pieces: ['गणपती', 'बप्पा', 'मोरया', '<unk>', 'पुढच्या', 'वर्षी',
 'लवकर', 'या', '<unk>']
               UNK tokens: 2 at positions [3, 8]

  -- vocab_size=800 --
     trained vocab_size: 800 (requested 800)
     SINDHI   | tokens=  8 | words=5 | fertility=1.600  (identical to 400)
     MARATHI  | tokens=  9 | words=7 | fertility=1.286  (identical to 400)
```


# Task 10 — The Compression Trade-off

## Introduction

Tokenizer compression refers to representing text using **fewer tokens**.
Lower token counts are desirable because language models process tokens rather than characters or words.

A common metric for measuring this is **fertility**:

```
fertility = tokens / words
```

Lower fertility means **better compression**.

However, improving compression can sometimes introduce other problems.
This experiment demonstrates a clear **compression trade-off** between two tokenizer configurations.

---

# Experiment Setup

Two tokenizers were trained on the same corpus but with different model types.

| Tokenizer   | Model Type | Vocabulary Size |
| ----------- | ---------- | --------------- |
| Tokenizer A | WordLevel  | 9               |
| Tokenizer B | BPE        | 66              |

Both tokenizers were evaluated using the same test sentences.

Test sentences:

```
machine learning models process tokens
भारत में कई भाषाएँ बोली जाती हैं
```

---

# Results

| Tokenizer | Total Tokens | Words | Fertility |
| --------- | ------------ | ----- | --------- |
| WordLevel | 2            | 12    | 0.167     |
| BPE       | 70           | 12    | 5.833     |

Token difference:

```
70 - 2 = 68 tokens
```

The WordLevel tokenizer produced **far fewer tokens**, making it the clear winner in terms of compression.

---

# What Improved?

Compression improved significantly when using the **WordLevel tokenizer**.

It produced:

* **2 tokens** instead of **70 tokens**
* A fertility of **0.167 vs 5.833**

This means the WordLevel tokenizer used **~35× fewer tokens**.

---

# What Got Worse?

The improvement in compression comes with a serious drawback.

The WordLevel tokenizer has a **very small vocabulary (9 tokens)**.
Because of this, most words in the test sentences are **not present in the vocabulary**.

When an unknown word appears, the tokenizer replaces it with a single special token:

```
[UNK]
```

This explains why entire sentences were represented by **one token**.

For example:

```
machine learning models process tokens
```

became:

```
[UNK]
```

This dramatically reduces token count but **destroys the information contained in the text**.

---

# Why the BPE Tokenizer Uses More Tokens

The BPE tokenizer splits words into **subword units** instead of replacing them with `[UNK]`.

For example:

```
tokenization → token ##ization
```

This allows BPE to handle **unseen words** while still preserving meaning.

However, this increases the number of tokens produced.

---

# The Trade-off

This experiment demonstrates a clear tension between two goals:

| Property                 | WordLevel | BPE   |
| ------------------------ | --------- | ----- |
| Compression              | Excellent | Worse |
| Handling unseen words    | Poor      | Good  |
| Information preservation | Poor      | Good  |
| Robustness               | Low       | High  |

So the trade-off can be summarized as:

```
compression vs robustness
```

The WordLevel tokenizer achieves better compression but loses important information when unknown words appear.

---

# Would This Be Used in Production?

In most real-world systems, the **BPE tokenizer would be preferred**.

Although it produces more tokens, it has several advantages:

* It can tokenize **unseen words**
* It preserves **more information**
* It generalizes better to new text

Modern large language models such as GPT-style models therefore use **subword tokenizers like BPE or SentencePiece** rather than WordLevel tokenizers.

---

# Key Insight

Better compression does not always mean better tokenization.

A tokenizer that produces extremely few tokens may simply be **collapsing unknown words into `[UNK]`**, which reduces token count but loses information.

A good tokenizer must balance **compression, robustness, and information preservation**.




# Task 11 — Can You Trust the Benchmark Numbers?

## Short answer

**Some numbers are completely trustworthy across reruns, and some are not.**

For the current benchmark implementation in [`src/abctokz/eval/benchmark.py`](../src/abctokz/eval/benchmark.py):

- **Token-derived metrics** are perfectly stable when you rerun the benchmark on the same saved tokenizer and the same corpus sample.
- **Timing-derived metrics** (`elapsed_seconds`, `throughput_sps`) vary between runs, which is normal.
- **One metric I would not trust even after many reruns is `round_trip_success_rate`**, because the benchmark is currently measuring a loaded tokenizer artifact, and in this repo the save/load path is known to lose pipeline components (Task 2). That means the number can be consistently wrong.

---

## What the benchmark runner actually does

Reading [`src/abctokz/eval/benchmark.py`](../src/abctokz/eval/benchmark.py):

1. It loads all corpus lines.
2. It samples `sample_size` lines with [`sample_lines`](../src/abctokz/data/sampling.py).
3. It loads each tokenizer artifact from disk with `Tokenizer.load(...)`.
4. It runs warmup passes.
5. It runs `encode_batch(sentences)` `timed_runs` times.
6. It **averages elapsed time across runs**.
7. It keeps **only the encodings from the last timed run** for all non-timing metrics.

The key lines are these:

```python
all_encodings = []
total_elapsed = 0.0
for _ in range(cfg.timed_runs):
    with timed() as t:
        encodings = tokenizer.encode_batch(sentences)
    total_elapsed += t["elapsed"]
    all_encodings = encodings  # keep last run for metrics

avg_elapsed = total_elapsed / cfg.timed_runs
decoded = [tokenizer.decode(enc.ids) for enc in all_encodings]
```

That design choice is the subtle one Task 11 is hinting at.

---

## Experiment setup

I created one fixed BPE tokenizer artifact and one fixed benchmark corpus, then ran the same `BenchmarkRunner` twice back-to-back.

### Corpus

10 multilingual lines repeated 250 times = **2,500 lines** total.

Examples included:
- `hello world`
- `नमस्ते दुनिया`
- `गणपती बप्पा मोरया`
- `आयो लाल सभई खुश`
- `hello नमस्ते world दुनिया`

### Tokenizer

- preset: `bpe_multilingual(vocab_size=300)`
- trained on the same corpus
- saved to a temporary artifact directory

### Benchmark config

```python
BenchmarkConfig(
    name='task11_repeatability',
    corpus_paths=[...],
    tokenizer_paths=[artifact_dir],
    sample_size=500,
    warmup_runs=2,
    timed_runs=12,
    languages=['mixed'],
)
```

Important detail: the sampled sentence set is deterministic because [`sample_lines`](../src/abctokz/data/sampling.py) uses `seed=42` by default.

---

## Actual benchmark outputs

### Run 1

```json
{
  "tokenizer_name": "bpe_tok",
  "language": "mixed",
  "n_sentences": 500,
  "throughput_sps": 28271.65,
  "mean_tokens_per_sentence": 21.31,
  "fertility": 5.2205,
  "unk_rate": 0.197841,
  "round_trip_success_rate": 0.0,
  "normalized_seq_length_ratio": 0.9952,
  "elapsed_seconds": 0.0177,
  "extra": {}
}
```

### Run 2

```json
{
  "tokenizer_name": "bpe_tok",
  "language": "mixed",
  "n_sentences": 500,
  "throughput_sps": 31728.46,
  "mean_tokens_per_sentence": 21.31,
  "fertility": 5.2205,
  "unk_rate": 0.197841,
  "round_trip_success_rate": 0.0,
  "normalized_seq_length_ratio": 0.9952,
  "elapsed_seconds": 0.0158,
  "extra": {}
}
```

### Direct comparison

| Metric | Run 1 | Run 2 | Stable? |
|---|---:|---:|---|
| `tokenizer_name` | `bpe_tok` | `bpe_tok` | Yes |
| `language` | `mixed` | `mixed` | Yes |
| `n_sentences` | 500 | 500 | Yes |
| `mean_tokens_per_sentence` | 21.31 | 21.31 | **Yes** |
| `fertility` | 5.2205 | 5.2205 | **Yes** |
| `unk_rate` | 0.197841 | 0.197841 | **Yes** |
| `round_trip_success_rate` | 0.0 | 0.0 | **Yes** |
| `normalized_seq_length_ratio` | 0.9952 | 0.9952 | **Yes** |
| `throughput_sps` | 28271.65 | 31728.46 | No |
| `elapsed_seconds` | 0.0177 | 0.0158 | No |

Timing drift between these two runs:

- `throughput_sps`: **+12.23%**
- `elapsed_seconds`: **-10.73%**

---

## Q1 — Which metrics are perfectly stable between runs?

### Perfectly stable in this experiment

- `mean_tokens_per_sentence`
- `fertility`
- `unk_rate`
- `round_trip_success_rate`
- `normalized_seq_length_ratio`
- plus metadata fields like `tokenizer_name`, `language`, `n_sentences`

### Why they are stable

These metrics depend only on:

1. the sampled sentence list
2. the loaded tokenizer artifact
3. deterministic `encode_batch()` / `decode()` behavior

All three are fixed here.

The corpus sample is deterministic because `sample_lines(..., seed=42)` is used implicitly.  So even if `sample_size < len(corpus)`, the same 500 lines are selected every time.

For a deterministic tokenizer, once the sentences are fixed, the encodings are identical. Therefore:

- token counts do not change
- word counts do not change
- `unk_id` counts do not change
- decoded strings do not change

So these metrics should be bit-for-bit stable across reruns.

---

## Q2 — Which metrics vary, and why is that acceptable?

### Metrics that vary

- `elapsed_seconds`
- `throughput_sps`

### Why variation is expected

These are wall-clock performance measurements. They are sensitive to:

- OS scheduler noise
- CPU frequency scaling / turbo boost
- cache warmth
- Python GC timing
- other processes on the machine
- branch prediction / memory locality effects

So small to moderate run-to-run variation is normal, even with warmups and fixed input.

The measured difference here was about 10–12%, which is believable for such short runs (`~16–18 ms`).  When the timed interval is that small, tiny absolute differences become large percentage swings.

### Why that is acceptable

Benchmarking throughput is not supposed to be perfectly repeatable at millisecond-scale resolution.  The purpose is comparative trend detection, not exact reproducibility to the second decimal place.

If two tokenizers differ by 2–3%, I would not trust that ranking. If one is 2× slower, I would.

---

## Q3 — Is there anything I would not trust, even after many runs?

## Yes: `round_trip_success_rate`

I would not trust this metric as currently implemented in the benchmark runner.

### Reason 1 — benchmark compares decoded text to raw sentences, not normalized sentences

The metric call in [`src/abctokz/eval/benchmark.py`](../src/abctokz/eval/benchmark.py) is:

```python
round_trip_success_rate(sentences, decoded)
```

But the metric itself supports a better mode:

```python
round_trip_success_rate(originals, decoded, normalized_originals=...)
```

That matters because the docs in [`docs/indic_support.md`](../docs/indic_support.md) explicitly say decode is intended to be lossless relative to the **normalized** form, not necessarily the raw form.

So the benchmark is already harsher than the documented contract.

### Reason 2 — benchmark loads saved artifacts, and save/load is currently lossy in this repo

From Task 2, we already established that `Tokenizer.save()` / `Tokenizer.load()` does **not** persist and restore the full pipeline correctly.  The loaded tokenizer loses normalizer and pre-tokenizer configuration.

That directly contaminates benchmark measurements, because `BenchmarkRunner` always benchmarks `Tokenizer.load(tok_path)`.

I verified the effect again with a small saved BPE artifact.  Examples from the loaded tokenizer:

```text
INPUT : 'hello world'
TOKENS: ['h', '##e', '##l', '##l', '##o', '## ', '##w', '##o', '##r', '##l', '##d']
DECODE: 'helloorld'

INPUT : 'नमस्ते दुनिया'
TOKENS: ['न', '##म', '##स', '##्', '##त', '##े', '## ', '##द', '##ु', '##न', '##ि', '##य', '##ा']
DECODE: 'नमस्तेुनिया'

INPUT : 'गणपती बप्पा मोरया'
DECODE: 'गणपतीप्पामोरया'
```

So `round_trip_success_rate = 0.0` is stable here, but it is not measuring a healthy tokenizer. It is measuring a **broken loaded artifact**.

### Broader implication

This is the most important trust lesson from Task 11:

> A metric can be perfectly stable across runs and still be untrustworthy, because it may be consistently measuring the wrong thing.

---

## Q4 — What would I change to make the benchmark more trustworthy?

### 1. Separate throughput measurement from intrinsic metric measurement

Right now the benchmark computes metrics from the **last timed run**.  I would change that to:

- run warmup passes
- run one untimed evaluation pass to compute `encodings` and `decoded`
- run separate timed passes that collect only elapsed time

That would make the benchmark easier to reason about:

- intrinsic metrics measure output quality
- timed runs measure speed only

These are different concerns and should not be coupled.

### 2. Pass normalized originals into `round_trip_success_rate`

The benchmark should compute:

```python
normalized_originals = [
    tokenizer._normalizer.normalize(s) if tokenizer._normalizer else s
    for s in sentences
]
round_trip_success_rate(sentences, decoded, normalized_originals=normalized_originals)
```

Or better, expose a public helper to normalize text without reaching into private attributes.

That would align the metric with the documented contract.

### 3. Fix the save/load boundary before trusting benchmark quality metrics

Since the benchmark loads tokenizer artifacts from disk, it inherits any serialization bug.  Until the full tokenizer pipeline is serialized and restored correctly, quality metrics from benchmarked artifacts are suspect.

### 4. Report variance for throughput, not just a mean

Instead of only `avg_elapsed`, the benchmark should report at least:

- mean
- min / max
- standard deviation
- maybe p50 / p95

For performance metrics, a single mean hides too much.

### 5. Make sample seed explicit in `BenchmarkConfig`

Sampling is currently reproducible because `sample_lines(..., seed=42)` uses a default seed, but this seed is not exposed in `BenchmarkConfig`.  That is okay for now, but it should be explicit in the saved config/report so reruns are auditable.

---

## Q5 — The subtle design choice: metrics from the last timed run only. Is it the right choice?

## My answer: acceptable for this deterministic codebase, but not the right design.

### Why it works today

For the current repository, `encode_batch(sentences)` is deterministic for a fixed loaded tokenizer and a fixed sentence list.  That means:

- every timed run produces identical encodings
- using the last run instead of the first or an average changes nothing

So in practice, the choice is harmless **today** for token-derived metrics.

### Why it is still the wrong abstraction

It mixes two logically separate tasks:

- measuring speed
- computing quality metrics

The benchmark currently assumes that the timed outputs are representative and deterministic. That assumption is unstated. If a future tokenizer were stateful, stochastic, or adaptive, using the last timed run for metrics would become incorrect.

Also, by computing decode-based quality metrics from the last timed run, the code quietly couples correctness reporting to the timing loop. That makes the benchmark harder to audit.

### Better design

The cleaner design is:

1. sample sentences once
2. compute encodings once for quality metrics
3. run repeated timed passes only for latency / throughput
4. report timing variance explicitly

That design is simpler, more explicit, and future-proof.

---

## Final judgment

### What I trust

- `mean_tokens_per_sentence`
- `fertility`
- `unk_rate`
- `normalized_seq_length_ratio`

I trust these to be perfectly stable for a fixed benchmark config and fixed saved artifact.

### What I trust only directionally

- `elapsed_seconds`
- `throughput_sps`

Useful for broad comparisons, not fine-grained ranking.

### What I do not trust yet

- `round_trip_success_rate`

Not because it is noisy, but because the current benchmark path can be **systematically wrong**:

- it compares against raw input instead of normalized input
- it benchmarks loaded artifacts, and the current artifact loader does not restore the full pipeline

So the right answer to Task 11 is not just "timing varies".  The deeper answer is:

> The benchmark is repeatable, but not all repeatable numbers are trustworthy.



# Task 15 — Find Something That Breaks

## Finding

An input made entirely of **unknown characters** gets silently erased by the default decode path.

The clearest reproductions are:
- punctuation-only input like `!!!`
- emoji-only input like `🙂🙂`
- mixed known + unknown input like `hello🙂नमस्ते`

This happens in both **BPE** and **Unigram**.

---

## Why this stood out

The public API and docs imply that decoding should reconstruct text, at least relative to the normalized form:

- [`src/abctokz/tokenizer.py`](../src/abctokz/tokenizer.py) describes decode as the inverse pipeline: `ids -> id_to_token -> decoder -> string`
- [`docs/indic_support.md`](../docs/indic_support.md) says: **"decoding is lossless relative to the normalized input"**

But if the input contains characters that are out of vocabulary, the default `Tokenizer.decode(...)` path drops them completely instead of preserving even a placeholder in the output.

---

## Reproduction

### Minimal setup

I trained a small tokenizer on a corpus that intentionally contains **no punctuation** and **no emoji**:

```python
from pathlib import Path
import tempfile

from abctokz import Tokenizer
from abctokz.config.defaults import bpe_multilingual, unigram_multilingual

corpus_lines = [
    "hello world",
    "नमस्ते दुनिया",
    "मराठी भाषा",
    "आयो लाल",
    "mixed hello नमस्ते world",
] * 40

with tempfile.TemporaryDirectory() as tmp:
    corpus_path = Path(tmp) / "corpus.txt"
    corpus_path.write_text("\n".join(corpus_lines), encoding="utf-8")

    config = bpe_multilingual(vocab_size=120)
    tok = Tokenizer.from_config(config)
    tok.train([str(corpus_path)], config)

    text = "!!!"
    enc = tok.encode(text)
    print(enc.ids)
    print(enc.tokens)
    print(tok.decode(enc.ids))
    print(tok.decode(enc.ids, skip_special_tokens=False))
```

Same behavior was then verified with `unigram_multilingual(vocab_size=120)`.

---

## Actual observed output

### BPE

```text
CASE: punctuation-only
  input        : '!!!'
  ids          : [0, 0, 0]
  tokens       : ['!', '##!', '##!']
  decode()     : ''
  decode(False): '<unk> <unk> <unk>'

CASE: emoji-only
  input        : '🙂🙂'
  ids          : [0, 0]
  tokens       : ['🙂', '##🙂']
  decode()     : ''
  decode(False): '<unk> <unk>'

CASE: mixed-known-plus-emoji
  input        : 'hello🙂नमस्ते'
  ids          : [48, 6, 10, 12, 0, 58, 28, 47, 20]
  tokens       : ['h', '##el', '##l', '##o', '##🙂', 'न', '##मस', '##्', '##ते']
  decode()     : 'hello नमस्ते'
  decode(False): 'hello <unk> नमस्ते'
```

### Unigram

```text
CASE: punctuation-only
  input        : '!!!'
  ids          : [0, 0, 0]
  tokens       : ['<unk>', '<unk>', '<unk>']
  decode()     : ''
  decode(False): '<unk> <unk> <unk>'

CASE: emoji-only
  input        : '🙂🙂'
  ids          : [0, 0]
  tokens       : ['<unk>', '<unk>']
  decode()     : ''
  decode(False): '<unk> <unk>'

CASE: mixed-known-plus-emoji
  input        : 'hello🙂नमस्ते'
  ids          : [1, 0, 3]
  tokens       : ['hello', '<unk>', 'नमस्ते']
  decode()     : 'hello नमस्ते'
  decode(False): 'hello <unk> नमस्ते'
```

---

## Observed vs expected

### Observed

- Unknown punctuation and emoji are encoded to ID `0` (`<unk>`).
- `Tokenizer.decode(ids)` defaults to `skip_special_tokens=True`.
- During decode, `<unk>` is treated like a special token and removed.
- Result: unknown user input disappears completely.

### Expected

At minimum, unknown content should not be silently erased.

Reasonable expected behaviors would be one of these:
- decode back to explicit placeholders like `<unk> <unk>`
- preserve the original unknown surface form via a byte/char fallback strategy
- document clearly that unknown content is lossy and excluded by default

The current behavior is the worst of the three because it hides data loss.

---

## Root cause in code

### 1. Top-level tokenizer strips anything that looks like `<...>`

In [`src/abctokz/tokenizer.py`](../src/abctokz/tokenizer.py), `decode()` defaults to `skip_special_tokens=True` and then removes not only configured special tokens, but also **any token string that starts with `<` and ends with `>`**.

That catches `<unk>` too.

### 2. The low-level decoders codify the same assumption

[`src/abctokz/decoders/word_decoder.py`](../src/abctokz/decoders/word_decoder.py) and [`src/abctokz/decoders/subword_decoder.py`](../src/abctokz/decoders/subword_decoder.py) also treat angle-bracket tokens as skippable "special" tokens.

### 3. The tests currently lock this in

[`tests/unit/test_decoders.py`](../tests/unit/test_decoders.py) explicitly asserts that `<unk>` should disappear when `skip_special_tokens=True`.

So this is not an accidental one-line mistake in a single layer. It is a consistent implementation choice across decoder code and tests.

---

## Classification

## Bug

This is a **bug in the public API behavior**, even though the internal decoder tests currently encode it as intended.

### Why I classify it as a bug

1. The top-level docs promise decode as the inverse path and claim normalized-form losslessness.
2. Silent deletion of unknown user content is a stronger failure than ordinary lossy tokenization.
3. `skip_special_tokens=True` should skip true control symbols like `<s>` or `</s>`, not unknown user content.
4. The failure is externally visible and surprising: `decode(encode("🙂🙂")) == ""`.

If the library wanted to define this as a limitation instead, the docs would need to say so clearly. Right now they say the opposite.

---

## Minimal workaround

### Workaround 1

Call:

```python
tokenizer.decode(ids, skip_special_tokens=False)
```

That at least exposes the loss as `<unk>` markers instead of silently deleting it.

### Limitation of the workaround

This still does **not** recover the original punctuation or emoji. It only makes the failure visible.

### Workaround 2

Train on representative data that actually includes the punctuation / emoji / symbols you expect in production.

For punctuation specifically, a stronger config would also include a punctuation-isolating pre-tokenizer so commas and exclamation marks become standalone units rather than glued word suffixes.

---

## Minimal fix

The smallest safe fix is:

- in `Tokenizer.decode(...)`, when `skip_special_tokens=True`, skip only tokens present in `self._special_tokens`
- do **not** generically drop every angle-bracket token
- keep `<unk>` in the output unless the user explicitly configured it as a special token

That would change the bad failure mode from:

```text
decode(encode('🙂🙂')) == ''
```

to at least:

```text
decode(encode('🙂🙂')) == '<unk> <unk>'
```

That is still lossy, but it is honest and debuggable.

A larger, more robust fix would be byte fallback or explicit unknown-character preservation, but that is beyond a minimal Task 15 scope.

---

## Why this case is useful

This edge case matters in real systems because punctuation, emoji, and symbols are not rare noise anymore. They appear in:

- chat and social data
- user reviews
- OCR or scraped text
- mixed-script content

Silently deleting them can change meaning and makes debugging downstream text issues much harder.

---

## Short conclusion

The most convincing break I found is:

```text
Tokenizer.decode(Tokenizer.encode('!!!').ids) == ''
Tokenizer.decode(Tokenizer.encode('🙂🙂').ids) == ''
```

That is a **bug**, not just a limitation, because the library documents decode as a meaningful inverse path and claims normalized-form losslessness, while the default decode behavior silently erases unknown user input.



# Task 16 — Is This Ready for Production?

**Scenario:** Deploy `abctokz` as the tokenizer for a Hindi+English text preprocessing pipeline
handling millions of documents per day.

---

## Three Reasons to Feel Confident

### 1. Deterministic, integrity-checked artifacts with schema versioning

Every trained tokenizer produces the same vocabulary given the same corpus and seed. All three
trainers (`BPETrainer`, `UnigramTrainer`, `WordLevelTrainer`) accept a `seed` parameter in their
config schemas (`src/abctokz/config/schemas.py`), and the property test suite verifies this
explicitly:

```
tests/property/test_determinism.py::TestDeterminism::test_bpe_encode_deterministic
tests/property/test_determinism.py::TestDeterminism::test_wordlevel_encode_deterministic
```

When saving, `Tokenizer.save()` (`src/abctokz/tokenizer.py`, lines ~315–340) computes a SHA-256
checksum of `vocab.json` via `src/abctokz/utils/hashing.py::sha256_file()` and writes it into
`manifest.json`. On `load()`, the schema version is checked immediately and raises
`SchemaVersionError` if mismatched. This means:

- Artifact corruption is detectable (checksum in manifest).
- Cross-version incompatibility surfaces immediately, not as silent wrong behavior.
- Reproduced training runs are byte-identical, so you can audit or retrain any artifact.

This is production-grade artifact management.

---

### 2. Actionable, typed exception hierarchy — no silent failures at the library boundary

`src/abctokz/exceptions.py` defines a layered hierarchy with specific, named exception classes:

| Exception | When raised |
|---|---|
| `SchemaVersionError` | artifact schema mismatch on `load()` |
| `SerializationError` | missing manifest file, unknown model_type |
| `VocabError` → `UnknownTokenError` | token not in vocabulary |
| `TrainingError` | training produces invalid state |
| `ConfigError` | invalid or inconsistent configuration |

Every exception carries a specific human-readable message:

```python
# exceptions.py
class SchemaVersionError(SerializationError):
    def __init__(self, found: str, expected: str) -> None:
        super().__init__(
            f"Incompatible schema version: found '{found}', expected '{expected}'."
            " Please retrain or migrate the artifact."
        )
```

No hot-path code raises bare `Exception`. Pydantic 2.x validates all configs at construction time
(`src/abctokz/config/schemas.py`), so misconfigured tokenizers fail loudly at setup rather than
producing wrong output silently. Structured logging via `src/abctokz/utils/logging.py` uses module
namespaces (`logging.getLogger("abctokz.*")`) and is safe to call multiple times.

In a real deployment, these properties mean: if something goes wrong, your monitoring can catch a
named exception and know exactly what it is.

---

### 3. Memory-safe streaming training + correct Devanagari normalization

Training uses Python generators throughout. The corpus iterator in `Tokenizer.train()`
(`src/abctokz/tokenizer.py`) and `src/abctokz/data/streaming.py::stream_shards()` never load the
full corpus into memory:

```python
def _corpus_iter():
    for path in corpus_paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                ...
                yield line
```

`stream_shards()` streams in sorted shard order, making training on multi-billion-token datasets
feasible with constant memory overhead.

The normalization preset used for production (`multilingual_shared_normalizer()` in
`src/abctokz/config/defaults.py`) deliberately uses **NFC, not NFKC**:

```python
# defaults.py
DevanagariNormalizerConfig(nfc_first=True, strip_zero_width=False)
```

The comment in the same file explains why: "NFKC would mangle some Devanagari combining marks."
This is the correct choice. NFKC collapses some Devanagari vowel markers in a way that is visually
invisible but textually wrong. A library that gets this right in its default preset can be trusted
with production Hindi/Marathi text.

---

## Three Reasons to Be Hesitant

### Gap 1 — `save()` silently drops the entire normalization + pretokenization pipeline [CRITICAL]

**Evidence:** `src/abctokz/tokenizer.py`, `save()` method, lines ~315–320:

```python
# We reconstruct a minimal config dict from what we know
model_type = self._infer_model_type()
config_data: dict[str, object] = {"model_type": model_type, "schema_version": SCHEMA_VERSION}
save_json(config_data, out / CONFIG_FILENAME)
```

Only two fields are written. `load()` (lines ~360–411) reconstructs only the model and decoder.
The `normalizer` and `pretokenizer` fields on the returned tokenizer are `None`.

This means: a tokenizer loaded from disk produces **different encodings** for the same text
compared to the same tokenizer immediately after training. The loaded version uses no whitespace
normalization, no DevanagariAwarePreTokenizer, no script-boundary splitting — just raw model lookup.

**Why the tests don't catch it:** `tests/integration/test_train_save_load.py` trains, saves,
loads, then checks:

```python
enc = loaded.encode("hello world")
assert len(enc.ids) > 0
decoded = loaded.decode(enc.ids)
assert isinstance(decoded, str)
```

It only checks that *something* comes back. It never checks
`loaded.encode(text) == trained.encode(text)`. So the test suite locks in the broken behavior.

**Production impact:** Any model downstream trained on the pre-tokenized output of the in-memory
tokenizer will receive differently-tokenized input at inference time (when the tokenizer is loaded
from disk). This invalidates all fertility numbers, benchmarks, and embedding alignments measured
on the in-memory version.

---

### Gap 2 — `decode()` silently erases unknown characters, producing empty strings without warning

**Evidence:** `src/abctokz/tokenizer.py`, `decode()` method, lines ~176–184:

```python
if skip_special_tokens:
    special_strs = set(self._special_tokens.keys())
    # Also skip tokens that look like <special>
    tokens = [
        t for t in tokens
        if t and not (t in special_strs or (t.startswith("<") and t.endswith(">")))
    ]
```

Any character that falls outside the vocabulary gets assigned `<unk>` by the model. `<unk>` starts
with `<` and ends with `>`, so this filter removes it. The result:

```python
decode(encode("!!!"))  # → ""
decode(encode("🙂🙂")) # → ""
```

Verified experimentally with both BPE and Unigram at `vocab_size=120`. Neither the `Encoding`
object nor the decode return value signals that data was lost. The `special_tokens_mask` in
the `Encoding` is `0` for `<unk>` tokens (because `<unk>` is not in `self._special_tokens`
unless explicitly registered), so even inspecting the mask doesn't help.

**Production impact:** In a text pipeline processing millions of documents, any document
containing only punctuation, emoji, symbols, or scripts not in the training corpus will silently
produce an empty string. Downstream systems (search indexes, classifiers, embeddings) would receive
empty strings with no indication that the input was non-empty. This is a data-loss bug, not a
limitation.

**The `UnknownTokenError` exception in `exceptions.py` exists but is never raised** — it is
defined, has a specific message, but no code path in the encode/decode pipeline throws it.

---

### Gap 3 — `encode_batch()` is a sequential Python loop; `decode()` rebuilds the inverse vocab on every call

**Evidence 1:** `src/abctokz/tokenizer.py`, `encode_batch()`:

```python
def encode_batch(self, texts: list[str]) -> list[Encoding]:
    return [self.encode(t) for t in texts]
```

A plain Python list comprehension. No thread pool, no `concurrent.futures`, no multiprocessing,
no Rust/C extension, no vectorized path. The GIL is held throughout.

**Evidence 2:** `src/abctokz/tokenizer.py`, `decode()`:

```python
vocab = self._model.get_vocab()
inv_vocab = {v: k for k, v in vocab.items()}
```

Every single `decode()` call pays O(vocab_size) to rebuild the inverse mapping from scratch. For
a vocabulary of 32,000 tokens, this is 32,000 dict insertions per decode call. There is no cached
`_inv_vocab` field, no lazy initialization, no `lru_cache`.

Similarly, `id_to_token()` has the same pattern.

**Production impact:** In Task 11, the measured throughput was ~500 sentences/second on a small
corpus with `vocab_size=300`. At 1M–100M documents/day, single-threaded Python at 500 sps =
43.2M sentences/day — barely sufficient at the low end of "millions," and wholly insufficient
for large-scale deployments. Parallelism requires either spawning separate processes (with full
model copies) or building a batched C extension. Neither is supported or documented.

There are also no thread-safety guarantees anywhere in the library. Multiple threads calling
`encode()` concurrently on the same tokenizer instance would be safe in practice (no instance state
is mutated during encode), but this is not guaranteed by any documented contract or test.

---

## Priority Ranking of Gaps

| Priority | Gap | Why |
|---|---|---|
| **1 (Fix first)** | `save()`/`load()` drops normalizer + pretokenizer | Causes *silent behavioral divergence* between training and inference. Every deployed artifact is broken by default. Impossible to catch without deep inspection. |
| **2** | `decode()` silently erases unknown chars | Causes *silent data loss* in the output pipeline. The `UnknownTokenError` exception exists but is never raised. One-line fix available. |
| **3** | Sequential `encode_batch()` + per-call inv_vocab rebuild | Performance concern for high-scale deployment, but not a correctness failure. Fixable with a cached `_inv_vocab` property (small change) and documented advice to use `multiprocessing`. |

**The single most important thing to fix first:** Gap 1 — the `save()`/`load()` pipeline truncation.

Rationale: You can work around the `decode()` bug by passing `skip_special_tokens=False` and
inspecting for `<unk>`. You can work around the throughput ceiling with process-level parallelism.
But there is no workaround for Gap 1 that doesn't require re-engineering the save/load process
manually every time you operationalize a new tokenizer version. It corrupts the fundamental
contract of the library: "train once, deploy anywhere from saved artifact."




# Task 17 — One Small Improvement

## Problem

The CLI command:

```
abctokz train
```

accepts a `--min-freq` parameter:

```
--min-freq INTEGER   Minimum token frequency
```

Users expect this flag to control **token filtering during training**.

However, when using **inline CLI mode** (passing `--corpus`, `--model`, etc. instead of `--config`), the provided value was **not applied to the trainer configuration**.

Example command:

```bash
abctokz train \
  --corpus data.txt \
  --model wordlevel \
  --min-freq 3 \
  --output artifacts/test
```

Expected behavior:

Tokens appearing fewer than **3 times** should be filtered.

Actual behavior:

The tokenizer continued using the **default `min_frequency` value**, ignoring the CLI override.

---

## Root Cause

`TokenizerConfig` and its nested trainer configs are **frozen Pydantic models**.

The original implementation attempted to mutate the trainer configuration in-place:

```python
tok_config.trainer.min_frequency = min_frequency
```

Because the config objects are immutable, this mutation **did not reliably update the configuration** used by the trainer.

---

# Fix

Instead of mutating the trainer configuration, create a **new updated config** using `model_copy()`.

This preserves immutability while correctly applying the CLI value.

### Code Change

File:

```
src/abctokz/cli/train.py
```

```diff
+ # Apply CLI min_frequency override if trainer supports it.
+ # TokenizerConfig and TrainerConfig models are frozen,
+ # so we must create updated copies instead of mutating.
+ if isinstance(tok_config.trainer, (WordLevelTrainerConfig, BPETrainerConfig)):
+     tok_config = tok_config.model_copy(
+         update={
+             "trainer": tok_config.trainer.model_copy(
+                 update={"min_frequency": min_frequency}
+             )
+         }
+     )
```

---

# Why This Fix Is Correct

This change is minimal and safe because:

* Only **CLI configuration wiring** is modified.
* No changes to training algorithms.
* Preserves the **immutable config design**.
* Only affects **inline CLI mode**.
* `--config` YAML mode continues working unchanged.

---

# Evidence (Before vs After)

Test corpus:

```
data/test_freq.txt
```

```
apple apple apple apple
banana banana
grape
```

Token frequencies:

| token  | count |
| ------ | ----- |
| apple  | 4     |
| banana | 2     |
| grape  | 1     |

---

## Expected Behavior

Using:

```bash
--min-freq 3
```

Only tokens with frequency **≥ 3** should remain.

Expected vocabulary:

```
apple
```

---

# Before Fix

Command:

```bash
abctokz train \
  --corpus data/test_freq.txt \
  --model wordlevel \
  --vocab-size 50 \
  --min-freq 3 \
  --output artifacts/test_before
```

Result:

```json
{
  "<unk>": 0,
  "apple": 1,
  "banana": 2
}
```

`banana` incorrectly appears even though its frequency is **2 (<3)**.

---

# After Fix

Command:

```bash
abctokz train \
  --corpus data/test_freq.txt \
  --model wordlevel \
  --vocab-size 50 \
  --min-freq 3 \
  --output artifacts/test_after
```

Result:

```json
{
  "<unk>": 0,
  "apple": 1
}
```

`banana` and `grape` are correctly filtered out.

---

# Artifacts Used

```
artifacts/test_before/vocab.json
artifacts/test_after/vocab.json
```

These demonstrate that the CLI flag now correctly controls token frequency filtering.




# Task 18 — Why This Change and Not a Bigger One?

## What I fixed (Task 17 recap)

I fixed a CLI contract mismatch: `abctokz train --min-freq` was accepted in inline mode but not reliably applied because the code attempted to mutate a frozen Pydantic config.

## How localized is the change?

- Touched:
  - `src/abctokz/cli/train.py` (inline-mode config construction only)
- Left alone:
  - All trainers (`src/abctokz/trainers/*`)
  - All models (`src/abctokz/models/*`)
  - Serialization/artifacts format
  - Benchmarking/eval code
  - YAML (`--config`) path behavior

In other words: no algorithm changes, just correct plumbing of a CLI flag into an existing config.

## Risk assessment

**What could break?**

- Inline training runs that rely on the _previously incorrect_ behavior (i.e., passing `--min-freq` but still getting the default `min_frequency=2`).

**What should not break?**

- Existing YAML-based training runs (`--config`), because those configs already specify `min_frequency` explicitly.
- Unigram training, because Unigram’s trainer config does not use `min_frequency`; the fix intentionally applies only to WordLevel and BPE.

**Why risk is low**

- The change uses Pydantic’s supported immutable-update mechanism (`model_copy(update=...)`) instead of in-place mutation.
- It is gated by trainer type (`WordLevelTrainerConfig`, `BPETrainerConfig`), so it cannot accidentally affect other trainer families.

## Expected impact

- **Who benefits:** anyone using the CLI in inline mode (hackathon participants, benchmark runners, quick experiments).
- **How they benefit:** `--min-freq` now actually does what the help text promises, making experiments reproducible and preventing “why is this token still in the vocab?” confusion.

## Bigger refactor that might be better in principle (but wrong here)

A larger refactor could:

- Centralize CLI→config overrides for all flags (vocab size, special tokens, normalization presets, etc.).
- Add shared validation like “these flags are only meaningful for certain model types”.
- Add tests around CLI flag wiring.

That’s likely a cleaner architecture long-term, but it’s the wrong move for this task because:

- It would touch multiple modules and increase surface area.
- It risks introducing new behavior changes unrelated to `--min-freq`.
- Task 18 is explicitly about tradeoffs: here, correctness with minimal blast radius beats “clean up everything”.

## Tradeoff summary

I kept the change small because this bug is a simple wiring issue with a clear correct behavior, and the safest fix is the smallest one that restores the CLI’s documented contract without changing training algorithms or artifact formats.





# Task 19 — BPE vs Unigram vs WordLevel: what’s actually different?

## Setup

Train all 3 models on the same corpus + same vocab size:


Then compare encodings + vocab stats:


Script used for the 5-input comparison:
- Default 5 inputs (when `--inputs` is not provided):
	- `hello world`
	- `antidisestablishmentarianism is hard`
	- `नमस्ते दुनिया`
	- `ॐ भूर्भुवः स्व: तत्सवितुर्वरेण्यं`
	- `hello नमस्ते world`


```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _force_utf8_stdout() -> None:
	if hasattr(sys.stdout, "reconfigure"):
		sys.stdout.reconfigure(encoding="utf-8")
	if hasattr(sys.stderr, "reconfigure"):
		sys.stderr.reconfigure(encoding="utf-8")


def _ensure_src_on_path() -> None:
	try:
		import abctokz  # type: ignore[import-not-found]  # noqa: F401
		return
	except Exception:
		repo_root = Path(__file__).resolve().parents[1]
		src_dir = repo_root / "src"
		if src_dir.exists():
			sys.path.insert(0, str(src_dir))


def _short_tokens(tokens: list[str], max_items: int = 14) -> str:
	if len(tokens) <= max_items:
		return " ".join(tokens)
	return " ".join(tokens[:max_items]) + f" … ({len(tokens)} toks)"


def _vocab_stats(name: str, vocab: dict[str, int]) -> dict[str, int]:
	# simple heuristics: whole-word vs subword-ish vs single-char-ish
	n = len(vocab)
	specials = sum(1 for t in vocab if t.startswith("<") and t.endswith(">"))

	if name == "wordlevel":
		single_char = sum(1 for t in vocab if len(t) == 1 and not (t.startswith("<") and t.endswith(">")))
		multi = n - specials - single_char
		return {"vocab": n, "special": specials, "single_char": single_char, "multi_char": multi}

	if name == "bpe":
		cont = 0
		cont_single = 0
		cont_multi = 0
		noncont_single = 0
		noncont_multi = 0

		for token in vocab:
			if token.startswith("<") and token.endswith(">"):
				continue
			is_cont = token.startswith("##")
			base = token[2:] if is_cont else token
			if is_cont:
				cont += 1
				if len(base) == 1:
					cont_single += 1
				else:
					cont_multi += 1
			else:
				if len(base) == 1:
					noncont_single += 1
				else:
					noncont_multi += 1

		return {
			"vocab": n,
			"special": specials,
			"continuation(##)": cont,
			"cont_single": cont_single,
			"cont_multi": cont_multi,
			"single_char": noncont_single,
			"multi_char": noncont_multi,
		}

	# unigram
	single = sum(1 for t in vocab if len(t) == 1 and not (t.startswith("<") and t.endswith(">")))
	multi = n - specials - single
	return {"vocab": n, "special": specials, "single_char": single, "multi_char": multi}


def main() -> int:
	_force_utf8_stdout()
	_ensure_src_on_path()

	from abctokz.tokenizer import Tokenizer  # type: ignore[import-not-found]

	ap = argparse.ArgumentParser(description="Task 19: compare WordLevel vs BPE vs Unigram")
	ap.add_argument(
		"--root",
		default=str(Path("artifacts") / "task19"),
		help="Root directory containing wordlevel/, bpe/, unigram/",
	)
	ap.add_argument(
		"--inputs",
		default="",
		help="Optional path to a utf-8 text file with one input per line.",
	)

	args = ap.parse_args()

	root = Path(args.root)
	paths = {
		"wordlevel": root / "wordlevel",
		"bpe": root / "bpe",
		"unigram": root / "unigram",
	}

	missing = [k for k, p in paths.items() if not p.exists()]
	if missing:
		print("Missing model artifacts:")
		for k in missing:
			print(f"- {k}: {paths[k]}")
		print("Train first:")
		print("  python -X utf8 task-scripts\\task_19_train_all.py")
		return 2

	toks = {k: Tokenizer.load(str(p)) for k, p in paths.items()}

	if args.inputs:
		inp_path = Path(args.inputs)
		lines = [ln.strip("\n") for ln in inp_path.read_text(encoding="utf-8").splitlines()]
		inputs = [ln for ln in lines if ln.strip()]
	else:
		inputs = [
			"hello world",
			"antidisestablishmentarianism is hard",
			"नमस्ते दुनिया",
			"ॐ भूर्भुवः स्व: तत्सवितुर्वरेण्यं",
			"hello नमस्ते world",
		]

	print("Task 19: side-by-side encodings")
	print(f"root: {root}")
	print()

	# vocab stats
	print("Vocab stats (rough)")
	for name, tok in toks.items():
		stats = _vocab_stats(name, tok.get_vocab())
		stats_str = ", ".join(f"{k}={v}" for k, v in stats.items())
		print(f"- {name}: {stats_str}")

	print("\nEncodings")
	for i, text in enumerate(inputs, start=1):
		print(f"\n[{i}] input: {text}")
		for name, tok in toks.items():
			enc = tok.encode(text)
			line = _short_tokens(enc.tokens)
			unk_id = tok.get_vocab().get("<unk>")
			if unk_id is None:
				unk_id = 0
			unk = sum(1 for tid in enc.ids if tid == unk_id)
			print(f"  {name:<9} unk#={unk:<3} {line}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
```

</details>

## Side-by-side tokenization 

Pick at least 5 inputs:
- easy English
- complex English
- simple Hindi
- complex Hindi
- mixed script


```text


Encodings

[1] input: hello world
	wordlevel unk#=1   <unk>
	bpe       unk#=1   h ##e ##l ##l ##o ##  ##w ##o ##r ##l ##d
	unigram   unk#=1   hello <unk> world

[2] input: antidisestablishmentarianism is hard
	wordlevel unk#=1   <unk>
	bpe       unk#=2   a ##n ##t ##i ##d ##i ##s ##e ##s ##t ##a ##b ##l ##i … (36 toks)
	unigram   unk#=6   antidisestab lishmentarianism <unk> is <unk> <unk> <unk> <unk> <unk>

[3] input: नमस्ते दुनिया
	wordlevel unk#=1   <unk>
	bpe       unk#=2   न ##म ##स ##् ##त ##े ##  ##द ##ु ##न ##ि ##य ##ा
	unigram   unk#=1   नमस्ते <unk> दुनिया

[4] input: ॐ भूर्भुवः स्व: तत्सवितुर्वरेण्यं
	wordlevel unk#=1   <unk>
	bpe       unk#=3   ॐ ##  ##भ ##ू ##र ##् ##भ ##ु ##व ##ः ##  ##स ##् ##व … (33 toks)
	unigram   unk#=24  ॐ <unk> भूर्भुवः <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <un  nk> <unk> <unk> … (26 toks)

[5] input: hello नमस्ते world
	wordlevel unk#=1   <unk>
	bpe       unk#=2   h ##e ##l ##l ##o ##  ##न ##म ##स ##् ##त ##े ##  ##w … (18 toks)
	unigram   unk#=7   hello <unk> नमस्ते <unk> <unk> <unk> <unk> <unk> <unk>
```



---
 n  
```text
Vocab stats (rough)
- wordlevel: vocab=30, special=1, single_char=2, multi_char=27
- bpe: vocab=187, special=1, continuation(##)=135, single_char=71, other=-20
- unigram: vocab=200, special=1, single_char=55, multi_char=144
```

- WordLevel produced only `<unk>` for every input. This happened because the learned vocabulary is tiny (30 total) and word-level models need exact token matches; anything OOV falls back to `<unk>`.
- BPE segmented everything into mostly character-ish pieces with lots of `##` continuations (e.g., `h ##e ##l...`). So it “always has a way to spell it”, but often over-segments when data/vocab are small.
- Unigram sometimes keeps larger chunks (e.g., `antidisestab` + `lishmentarianism`), but for the Hindi/mantra-like inputs it emitted many `<unk>` pieces, especially on the long mixed-symbol line.

---

## What dominates the vocabulary?

From the “Vocab stats” output:
- WordLevel should skew toward whole words
- BPE should show lots of `##` continuation pieces (subwords/characters)
- Unigram should be a mix of single chars + multi-char pieces


- WordLevel: `multi_char` dominates (27/30). This is basically a small set of whole tokens.
- BPE: `continuation(##)` dominates (135/187), and most of those are multi-char continuations (`cont_multi=87`). This indicates the vocab is mainly subword pieces, not whole words.
- Unigram: mixed inventory (55 single chars, 144 multi-char). It keeps a balance of character coverage + larger pieces.

---

## Which model would I choose?

(a) Low-resource language:
- Prefer Unigram (or BPE) over WordLevel: it can still tokenize via smaller pieces even when you can’t memorize many full words.

(b) Agglutinative language (Hindi / Finnish):
- Unigram or BPE: both can form morpheme-like pieces; WordLevel tends to explode OOV.
- In these outputs, BPE over-segmented heavily into characters; Unigram kept some larger chunks but still struggled on rare symbols.

(c) Consistent boundaries across languages:
- BPE is a common choice for multilingual consistency because it shares subword pieces across scripts and doesn’t depend on word boundaries.
- WordLevel only works if your boundary rules are consistent and your vocab is large enough to cover most tokens.

---



# Task 20 - Tokenization (Ways to split text so we can optimize 

Tokenization is the process of breaking text into smaller pieces called **tokens** so a computer can process language. Think of it as chopping a sentence into manageable parts that a machine can analyze.

Tokens can be words, subwords, character or even punctuation, symbols.

At first, this sounds simple. For example:

Original text:
"I love AI"

Tokens:
["I", "love", "AI"]

However, real-world language makes tokenization much more complex than simply splitting by spaces.

Consider the word **“tokenization.”** Instead of treating it as a single word, many modern tokenizers break it into smaller meaningful parts like **“token”** and **“##ization.”** This allows models to reuse patterns across many related words such as *token*, *tokenize*, and *tokenization*. These smaller pieces are called **subword tokens**.

Another challenge appears when working with **multiple languages**. English separates words with spaces, but many languages follow different rules. For example, scripts like **Hindi (Devanagari)** use combining characters, and some languages do not use spaces consistently at all. A tokenizer must still split text into useful pieces while preserving meaning.

During experimentation of Task 10, I observed that different tokenizers behave very differently. A **WordLevel tokenizer** compressed entire sentences into a single token when the words were unknown, while a **BPE tokenizer** split them into many smaller subwords. This showed that tokenization is a trade-off between **compression and preserving information**.

Because of these challenges—handling word structure, unseen words, and different writing systems—tokenization is far more than just splitting text. It is a key step that determines how efficiently and accurately language models understand human language.