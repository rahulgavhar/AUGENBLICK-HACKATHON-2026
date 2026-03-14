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
