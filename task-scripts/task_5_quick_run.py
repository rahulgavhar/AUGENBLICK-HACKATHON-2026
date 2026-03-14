from __future__ import annotations

import sys
from pathlib import Path


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


def _force_utf8_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


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

    missing = [name for name, p in models.items() if not p.exists()]
    if missing:
        print("Missing artifacts. Train first:")
        print("  python -X utf8 task-scripts\\task_5_train_models.py")
        for name in missing:
            print(f"  - {name}: {models[name]}")
        return 2

    tokenizers = {name: Tokenizer.load(str(path)) for name, path in models.items()}

    # Prefer looking up <unk> id from vocab, but default to UNK_ID.
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

    print("Task 5 quick UNK check")
    print(f"unk_id={unk_id}")
    print()

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
