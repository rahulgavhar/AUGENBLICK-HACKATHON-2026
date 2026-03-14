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
        import abctokz  # noqa: F401
        return
    except Exception:
        repo_root = Path(__file__).resolve().parents[1]
        src_dir = repo_root / "src"
        if src_dir.exists():
            sys.path.insert(0, str(src_dir))


def _write_default_corpus(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            "hello world",
            "the quick brown fox jumps over the lazy dog",
            "antidisestablishmentarianism is a long word",
            "नमस्ते दुनिया",
            "यह एक सरल वाक्य है",
            "ॐ भूर्भुवः स्व: तत्सवितुर्वरेण्यं भर्गो देवस्य धीमहि",
            "hello नमस्ते world",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


def main() -> int:
    _force_utf8_stdout()
    _ensure_src_on_path()

    from abctokz.config.defaults import bpe_multilingual
    from abctokz.config.schemas import (
        BPEConfig,
        BPETrainerConfig,
        TokenizerConfig,
        UnigramConfig,
        UnigramTrainerConfig,
        WordLevelConfig,
        WordLevelTrainerConfig,
    )
    from abctokz.tokenizer import Tokenizer

    ap = argparse.ArgumentParser(
        description="Task 19: train WordLevel/BPE/Unigram on identical corpus + vocab size",
    )
    ap.add_argument(
        "--corpus",
        default=str(Path("data") / "task19_corpus.txt"),
        help="Path to training corpus (utf-8). Default: data/task19_corpus.txt",
    )
    ap.add_argument(
        "--vocab-size",
        type=int,
        default=200,
        help="Target vocab size for all three models.",
    )
    ap.add_argument(
        "--out",
        default=str(Path("artifacts") / "task19"),
        help="Output directory root (3 subfolders will be created).",
    )

    args = ap.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        _write_default_corpus(corpus_path)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # Use the same normalizer + pretokenizer wiring for all three by starting
    # from the multilingual preset.
    base = bpe_multilingual(vocab_size=args.vocab_size)
    normalizer_cfg = base.normalizer
    pretokenizer_cfg = base.pretokenizer

    cfg_wordlevel = TokenizerConfig(
        normalizer=normalizer_cfg,
        pretokenizer=pretokenizer_cfg,
        model=WordLevelConfig(vocab_size=args.vocab_size),
        trainer=WordLevelTrainerConfig(vocab_size=args.vocab_size, min_frequency=1),
    )
    cfg_bpe = TokenizerConfig(
        normalizer=normalizer_cfg,
        pretokenizer=pretokenizer_cfg,
        model=BPEConfig(vocab_size=args.vocab_size),
        trainer=BPETrainerConfig(vocab_size=args.vocab_size, min_frequency=1),
    )
    cfg_unigram = TokenizerConfig(
        normalizer=normalizer_cfg,
        pretokenizer=pretokenizer_cfg,
        model=UnigramConfig(vocab_size=args.vocab_size),
        trainer=UnigramTrainerConfig(vocab_size=args.vocab_size),
    )

    jobs = [
        ("wordlevel", cfg_wordlevel, out_root / "wordlevel"),
        ("bpe", cfg_bpe, out_root / "bpe"),
        ("unigram", cfg_unigram, out_root / "unigram"),
    ]

    print("Task 19 training")
    print(f"corpus: {corpus_path}")
    print(f"vocab_size: {args.vocab_size}")

    for name, cfg, out_dir in jobs:
        tok = Tokenizer.from_config(cfg)
        tok.train([str(corpus_path)], cfg)
        tok.save(str(out_dir))
        print(f"- saved {name}: {out_dir} (vocab={tok.get_vocab_size()})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
