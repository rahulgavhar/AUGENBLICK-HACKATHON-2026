from __future__ import annotations

import sys
from pathlib import Path


def _force_utf8_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def _section(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{title}\n{bar}")


def _write_english_corpus(path: Path) -> None:
    # English-only corpus on purpose, so Devanagari/emoji will be OOV.
    # Include full alphabet so BPE/Unigram can still segment rare English words.
    corpus = "\n".join(
        [
            "the quick brown fox jumps over the lazy dog",
            "pack my box with five dozen liquor jugs",
            "sphinx of black quartz judge my vow",
            "abcdefghijklmnopqrstuvwxyz",
            "hello world",
            "hello there general kenobi",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(corpus + "\n", encoding="utf-8")


def main() -> int:
    _force_utf8_stdout()

    _section("Task 5: train English-only tokenizers (WordLevel / BPE / Unigram)")

    from abctokz.config.defaults import english_basic_normalizer
    from abctokz.config.schemas import (
        BPEConfig,
        BPETrainerConfig,
        TokenizerConfig,
        UnigramConfig,
        UnigramTrainerConfig,
        WhitespacePreTokenizerConfig,
        WordLevelConfig,
        WordLevelTrainerConfig,
    )
    from abctokz.tokenizer import Tokenizer

    corpus_path = Path("data") / "task5_english_only.txt"
    _write_english_corpus(corpus_path)
    print(f"Corpus: {corpus_path}")

    out_root = Path("artifacts")
    out_root.mkdir(parents=True, exist_ok=True)

    normalizer_cfg = english_basic_normalizer()
    pretokenizer_cfg = WhitespacePreTokenizerConfig()

    # Keep vocab reasonably small but large enough to include alphabet pieces.
    vocab_size = 200

    configs: list[tuple[str, TokenizerConfig, Path]] = [
        (
            "wordlevel",
            TokenizerConfig(
                normalizer=normalizer_cfg,
                pretokenizer=pretokenizer_cfg,
                model=WordLevelConfig(vocab_size=vocab_size),
                trainer=WordLevelTrainerConfig(vocab_size=vocab_size, min_frequency=1),
            ),
            out_root / "task5_eng_wordlevel",
        ),
        (
            "bpe",
            TokenizerConfig(
                normalizer=normalizer_cfg,
                pretokenizer=pretokenizer_cfg,
                model=BPEConfig(vocab_size=vocab_size),
                trainer=BPETrainerConfig(vocab_size=vocab_size, min_frequency=1),
            ),
            out_root / "task5_eng_bpe",
        ),
        (
            "unigram",
            TokenizerConfig(
                normalizer=normalizer_cfg,
                pretokenizer=pretokenizer_cfg,
                model=UnigramConfig(vocab_size=vocab_size),
                trainer=UnigramTrainerConfig(vocab_size=vocab_size),
            ),
            out_root / "task5_eng_unigram",
        ),
    ]

    for name, cfg, out_dir in configs:
        _section(f"Training: {name}")
        tok = Tokenizer.from_config(cfg)
        tok.train([str(corpus_path)], cfg)
        tok.save(str(out_dir))
        print(f"Saved: {out_dir}")
        print(f"Vocab size: {tok.get_vocab_size()}")

    _section("Done")
    print("Next: run task-scripts/task_5_unk_checks.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
