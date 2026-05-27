"""score romanian gec with upstream errant_compare.

upstream `errant_parallel` (english) is bypassed: m2 files are emitted directly
from token-level alignment via utils.write_m2 over the romanian word
tokenizer. `errant_compare` itself is language-agnostic (set arithmetic on
edit spans in default mode), so it scores romanian m2 correctly.

container expectation: errant 3.x's `errant_compare` binary on PATH, or in a
location passed via bin_dir.
"""
import re
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from utils import normalize_romanian, word_tokenize, write_m2


_INSTALL_HINT = (
    "errant scoring skipped.\n"
    "to enable: install errant 3.x in the container so `errant_compare` is on PATH,\n"
    "or pass --errant_bin_dir pointing to its install directory."
)


def _write_tokenized(path: Path, lines: list[str]) -> None:
    """one sentence per line, tokenized via word_tokenize and space-joined.
    normalize_romanian is applied first to keep diacritics consistent."""
    with path.open("w", encoding="utf-8") as f:
        for s in lines:
            s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()
            tokens = word_tokenize(normalize_romanian(s))
            f.write(" ".join(tokens) + "\n")


def _parse_compare_output(text: str) -> dict | None:
    """parse stdout from `errant_compare`. assumes errant 3.x span-based
    correction block, where a header row of the form

        TP    FP    FN    Prec    Rec    F0.5

    is followed (after blank or "==..." divider lines) by a whitespace-
    separated data row with 6 numeric columns: int int int float float float.
    extracts the last three as precision, recall, f05.

    returns None if the expected layout is not found.
    """
    lines = text.splitlines()
    header_re = re.compile(r"^\s*TP\b.*\bFP\b.*\bFN\b.*\bF0\.5\b\s*$")
    for i, line in enumerate(lines):
        if not header_re.match(line):
            continue
        for j in range(i + 1, len(lines)):
            data = lines[j].strip()
            if not data or data.startswith("="):
                continue
            cols = data.split()
            if len(cols) < 6:
                return None
            try:
                return {
                    "precision": float(cols[3]),
                    "recall": float(cols[4]),
                    "f05": float(cols[5]),
                }
            except ValueError:
                return None
    return None


def errant_score(
    sources: list[str],
    hypotheses: list[str],
    references: list[str],
    work_dir: Path,
    keep_tmp: bool = False,
    bin_dir: str | None = None,
) -> dict | None:
    """score (sources, hypotheses) against (sources, references).
    returns {"precision", "recall", "f05", "n"} or None on failure.
    """
    assert len(sources) == len(hypotheses) == len(references), (
        f"length mismatch: src={len(sources)} hyp={len(hypotheses)} ref={len(references)}"
    )

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    src_path = work_dir / "src.txt"
    hyp_path = work_dir / "hyp.txt"
    ref_path = work_dir / "ref.txt"
    hyp_m2 = work_dir / "hyp.m2"
    ref_m2 = work_dir / "ref.m2"
    tmp_paths = [src_path, hyp_path, ref_path, hyp_m2, ref_m2]

    compare_bin = f"{bin_dir}/errant_compare" if bin_dir else "errant_compare"

    try:
        _write_tokenized(src_path, sources)
        _write_tokenized(hyp_path, hypotheses)
        _write_tokenized(ref_path, references)

        # emit m2 files manually using romanian token alignment.
        write_m2(src_path, hyp_path, hyp_m2, normalize=True)
        write_m2(src_path, ref_path, ref_m2, normalize=True)

        cmd = [compare_bin, "-hyp", str(hyp_m2), "-ref", str(ref_m2)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"errant_compare failed (returncode {proc.returncode})", file=sys.stderr)
            if proc.stderr:
                print("stderr:", proc.stderr.strip(), file=sys.stderr)
            print(_INSTALL_HINT, file=sys.stderr)
            return None

        parsed = _parse_compare_output(proc.stdout)
        if parsed is None:
            print("errant_compare output did not contain a recognizable TP/FP/FN block.", file=sys.stderr)
            print("--- raw stdout ---", file=sys.stderr)
            print(proc.stdout, file=sys.stderr)
            print("--- end ---", file=sys.stderr)
            return None
        parsed["n"] = len(sources)
        return parsed

    except FileNotFoundError as e:
        print(f"errant_compare binary not found: {e}", file=sys.stderr)
        print(_INSTALL_HINT, file=sys.stderr)
        return None
    except Exception as e:
        print(f"errant scoring failed: {type(e).__name__}: {e}", file=sys.stderr)
        print(_INSTALL_HINT, file=sys.stderr)
        return None
    finally:
        if not keep_tmp:
            for p in tmp_paths:
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
