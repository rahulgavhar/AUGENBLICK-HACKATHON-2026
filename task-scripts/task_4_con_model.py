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
