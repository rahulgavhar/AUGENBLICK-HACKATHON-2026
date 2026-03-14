from __future__ import annotations

import sys


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

    _section("Task 4 validation failure demos")

    from abctokz.config.schemas import (
        BPEConfig,
        TokenizerConfig,
        WordLevelTrainerConfig,
    )

    _section("Failure mode 1: invalid vocab_size (ge=1)")
    try:
        _ = BPEConfig(vocab_size=0)
    except Exception as exc:
        _show_exception("BPEConfig(vocab_size=0)", exc)

    _section("Failure mode 2: mismatched model + trainer types")
    try:
        _ = TokenizerConfig(
            model=BPEConfig(vocab_size=10),
            trainer=WordLevelTrainerConfig(vocab_size=10),
        )
    except Exception as exc:
        _show_exception(
            "TokenizerConfig(model=BPEConfig(...), trainer=WordLevelTrainerConfig(...))",
            exc,
        )

    _section("Optional: forbidden extra fields (extra='forbid')")
    try:
        _ = BPEConfig(vocab_size=10, made_up_field=123)  # type: ignore[call-arg]
    except Exception as exc:
        _show_exception("BPEConfig(..., made_up_field=123)", exc)

    _section("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
