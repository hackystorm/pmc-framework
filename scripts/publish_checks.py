# SPDX-License-Identifier: GPL-3.0-only

"""Publish checks.

Four gates run before anything is made public:

1. **English** -- the repository must be in English. Italian prose in the
   source, the documentation or the comments is a failure, because the
   project is for everyone who reads English.
2. **No personal data** -- no names, home paths, session identifiers or
   fragments of anybody's private life belong in a public repository.
3. **No secrets** -- no keys, tokens or credentials of any kind.
4. **Licences** -- the GPL text is present and intact, SPDX identifiers are
   declared, and the model licence exists.

Run it with:

    python3 scripts/publish_checks.py

It exits non-zero when a check fails, and prints every finding rather than
stopping at the first one.

The checks are deliberately static: literal substrings, fixed word lists and
token comparison. No regular expressions, so there is nothing to misread and
nothing to tune by accident.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()

TEXT_SUFFIXES = {".py", ".rs", ".md", ".toml", ".cff", ".yml", ".yaml", ".txt", ".cfg", ".ini"}
SKIP_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "target",
    "node_modules",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "build",
    "dist",
    "egg-info",
}

# Italian function words, checked as whole words. Words that also exist in
# English ("come", "solo", "ante") are deliberately absent, because a false
# positive in a publish gate is worse than a missed one.
#
# "del" was removed from the list for the same reason: it is Italian for "of
# the" and also the Python delete statement, so any file that removed a
# dictionary entry failed the gate on a statement in good English.
ITALIAN_WORDS = frozenset(
    {
        "anche", "chi", "della", "delle", "dello", "degli",
        "dei", "dentro", "dopo", "dove", "dunque", "ecco", "essere",
        "fare", "fatto", "fatta", "fuori", "già", "giorno", "hanno",
        "insieme", "mai", "mente", "molto", "nella", "nelle", "nello",
        "negli", "nei", "nel", "nulla", "ogni", "oppure", "perché", "percio",
        "piuttosto", "poi", "prima", "può", "quale", "quando", "quello",
        "questa", "queste", "questi", "questo", "qui", "ricordo", "ricordi",
        "sempre", "senza", "sono", "sogno", "sogni", "stati",
        "stato", "stesso", "suo", "troppo", "tutto",
        "unico", "vale", "verso", "vita", "vive", "voglio",
    }
)

# Characters that are typography or diagrams rather than another language.
ALLOWED_NON_ASCII = frozenset(
    "—–…‘’“”•→←↔×≥≤≈°§¶†‡«»′″€£¥©®™"
    "┌┐└┘├┤┬┴┼─│┃┏┓┗┛┣┫┳┻╋"
    "▲▼◀▶△▽◁▷"
)

# Literal fragments that must never appear in a public repository, matched
# anywhere inside a chunk. These are path fragments: each contains a separator,
# so it cannot occur inside an ordinary identifier, and a home directory always
# arrives embedded in one ("workspace=r\"C:\\Users\\...\""), which a prefix test
# misses. Both separators are listed because a POSIX-only list silently passes
# Windows paths, and Windows is one of the two CI platforms here.
FORBIDDEN_PATH_FRAGMENTS = (
    "/home/",
    "/root/",
    "/Users/",
    "\\home\\",
    "\\root\\",
    "\\Users\\",
    "C:\\Users\\",
)

# Private names and identifiers, matched as a chunk prefix only. A substring
# test would over-match: "ses_" is the session-id prefix and it also begins
# "sessions_", which this repository uses everywhere by design. The rule is the
# one the English check already states: a false positive in a publish gate is
# worse than a missed one.
FORBIDDEN_NAME_FRAGMENTS = (
    "ses_",
    "darkyeye",
    "dualbrain",
    "dossier",
)

# Whole words that must never appear in a public repository.
FORBIDDEN_WORDS = frozenset({"ALe", "Ellydi", "Ellidi", "EllidiAgent"})

# Machine-local binary formats. They hold absolute paths to one machine's
# filesystem by construction, and no text check can read them, so they need a
# suffix rule rather than a literal-fragment rule.
STRAY_BINARY_SUFFIXES = frozenset({".lnk", ".url", ".exe", ".dll", ".pdb", ".so", ".dylib"})

# Literal blocks that indicate a leaked private key.
SECRET_BLOCKS = (
    "-----BEGIN RSA PRIVATE KEY-----",
    "-----BEGIN PRIVATE KEY-----",
    "-----BEGIN OPENSSH PRIVATE KEY-----",
    "-----BEGIN EC PRIVATE KEY-----",
    "-----BEGIN DSA PRIVATE KEY-----",
    "-----BEGIN PGP PRIVATE KEY-----",
)

# Prefixes that indicate a leaked credential. Checked against whole tokens,
# so that a word like "risk-adjusted" cannot trigger them.
SECRET_PREFIXES = (
    "sk-",
    "ghp_",
    "gho_",
    "ghu_",
    "ghs_",
    "ghr_",
    "github_pat_",
    "AKIA",
    "ASIA",
    "AIza",
    "xoxb-",
    "xoxp-",
    "xoxa-",
    "xoxr-",
    "xoxs-",
    "glpat-",
)

# Words that, combined with an assignment and a quoted value, look like a
# hardcoded credential.
CREDENTIAL_WORDS = (
    "password",
    "passwd",
    "api_key",
    "api-key",
    "apikey",
    "secret_key",
    "secret-key",
    "secretkey",
    "access_token",
    "private_key",
)

# Files that are allowed to contain the forbidden fragments, because they
# are the tests that enforce the rule and therefore have to name it.
PERSONAL_CHECK_EXEMPT = frozenset(
    {
        "tests/test_companion.py",
        "tests/test_publish_checks.py",
    }
)

# The same rule for the English check. A test that asserts the Italian detector
# still fires has to name an Italian word, so without an exemption the gate
# forbids the tests that prove it works. The exemption is listed here, next to
# the fragment list, so it cannot be added quietly.
ENGLISH_CHECK_EXEMPT = frozenset(
    {
        "tests/test_companion.py",
        "tests/test_publish_checks.py",
    }
)

# Files that legitimately reference credential field names (api_key and
# friends) because they implement or test the BYOK layer. They read keys
# from the environment; they never store them.
SECRETS_CHECK_EXEMPT = frozenset(
    {
        "src/pmc/runtime/client.py",
        "src/pmc/runtime/providers.py",
        "tests/test_providers.py",
    }
)

# The phrases that identify the canonical GPL-3.0 text.
GPL_REQUIRED_PHRASES = (
    "GNU GENERAL PUBLIC LICENSE",
    "Version 3, 29 June 2007",
    "TERMS AND CONDITIONS",
    "END OF TERMS AND CONDITIONS",
)

PUNCTUATION = ".,;:!?()[]{}\"'`~*_<>|/\\+=&%$#@^"


def strip_punctuation(token: str) -> str:
    """Remove leading and trailing punctuation from a token."""
    return token.strip(PUNCTUATION)


def tokenize(text: str) -> list[str]:
    """Split text into lowercase words, without punctuation."""
    cleaned = text.replace("'", " ").replace("-", " ")
    return [strip_punctuation(word.lower()) for word in cleaned.split()]


def chunks(text: str) -> list[str]:
    """Split a line into whitespace-separated chunks.

    Chunks keep their internal punctuation, so a path such as
    ``/home/someone`` stays one chunk and can be compared as a prefix.
    """
    return [chunk.lstrip("\"'`") for chunk in text.split()]


def iter_files() -> list[Path]:
    files: list[Path] = []
    for path in sorted(REPO_ROOT.rglob("*")):
        if not path.is_file():
            continue
        if path.resolve() == SELF:
            continue
        if any(part in SKIP_DIRECTORIES for part in path.parts):
            continue
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {
            "LICENSE",
            "CODE_OF_CONDUCT",
            "Makefile",
        }:
            files.append(path)
    return files


def relative(path: Path) -> str:
    """The path relative to the repository root, with POSIX separators.

    The exempt-file sets are written with forward slashes, so the
    comparison must not depend on the platform's separator: on Windows the
    default separator is a backslash and the exemption would never match.
    """
    return path.relative_to(REPO_ROOT).as_posix()


def check_english(files: list[Path]) -> list[str]:
    """The repository must be in English."""
    findings: list[str] = []

    for path in files:
        if relative(path) in ENGLISH_CHECK_EXEMPT:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), start=1):
            for character in line:
                if ord(character) > 127 and character not in ALLOWED_NON_ASCII:
                    findings.append(
                        f"{relative(path)}:{number}: non-ASCII character {character!r}"
                    )
                    break

            for word in tokenize(line):
                if word in ITALIAN_WORDS:
                    findings.append(
                        f"{relative(path)}:{number}: possible Italian text ({word!r})"
                    )
                    break

    return findings


def check_personal_data(files: list[Path]) -> list[str]:
    """Nothing private may appear in a public repository."""
    findings: list[str] = []

    for path in files:
        if relative(path) in PERSONAL_CHECK_EXEMPT:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), start=1):
            for chunk in chunks(line):
                lowered = chunk.lower()
                for fragment in FORBIDDEN_PATH_FRAGMENTS:
                    # A path fragment is checked as a substring: it contains a
                    # separator, so it cannot hide inside an identifier, and the
                    # leaking form is an assignment rather than a bare path.
                    if fragment.lower() in lowered:
                        findings.append(
                            f"{relative(path)}:{number}: forbidden path fragment {fragment!r}"
                        )
                        break
                else:
                    for fragment in FORBIDDEN_NAME_FRAGMENTS:
                        # A name fragment is checked as a prefix, because a
                        # substring would match ordinary identifiers.
                        if lowered.startswith(fragment.lower()):
                            findings.append(
                                f"{relative(path)}:{number}: forbidden identifier {fragment!r}"
                            )
                            break
                    else:
                        continue
                break

            for word in tokenize(line):
                if word in {w.lower() for w in FORBIDDEN_WORDS}:
                    findings.append(
                        f"{relative(path)}:{number}: forbidden identifier {word!r}"
                    )
                    break

    return findings


def check_secrets(files: list[Path]) -> list[str]:
    """No credential may appear in a public repository."""
    findings: list[str] = []

    for path in files:
        if relative(path) in SECRETS_CHECK_EXEMPT:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), start=1):

            for block in SECRET_BLOCKS:
                if block in line:
                    findings.append(
                        f"{relative(path)}:{number}: private key block"
                    )
                    break

            for token in tokenize(line):
                for prefix in SECRET_PREFIXES:
                    if token.startswith(prefix.lower()):
                        findings.append(
                            f"{relative(path)}:{number}: credential prefix {prefix!r}"
                        )
                        break

            lowered = line.lower()
            for word in CREDENTIAL_WORDS:
                if word in lowered and "=" in line and ("'" in line or '"' in line):
                    findings.append(
                        f"{relative(path)}:{number}: possible hardcoded credential ({word!r})"
                    )
                    break

    return findings


def check_licences(files: list[Path]) -> list[str]:
    """The licences must be present, intact and declared."""
    findings: list[str] = []

    license_path = REPO_ROOT / "LICENSE"
    if not license_path.exists():
        findings.append("LICENSE: missing")
    else:
        text = license_path.read_text(encoding="utf-8")
        for phrase in GPL_REQUIRED_PHRASES:
            if phrase not in text:
                findings.append(f"LICENSE: missing the phrase {phrase!r}")
        if len(text) < 30000:
            findings.append("LICENSE: the text is too short to be the GPL")

    model_license = REPO_ROOT / "MODEL_LICENSE.md"
    if not model_license.exists():
        findings.append("MODEL_LICENSE.md: missing")
    elif "model weights" not in model_license.read_text(encoding="utf-8").lower():
        findings.append("MODEL_LICENSE.md: does not say what it covers")

    sources = list((REPO_ROOT / "src").rglob("*.py"))
    sources += list((REPO_ROOT / "tests").rglob("*.py"))
    sources += list((REPO_ROOT / "scripts").rglob("*.py"))
    sources += list((REPO_ROOT / "crates").rglob("*.rs"))
    for path in sources:
        head = path.read_text(encoding="utf-8")[:2000]
        if "SPDX-License-Identifier: GPL-3.0-only" not in head:
            findings.append(f"{relative(path)}: missing the SPDX licence identifier")

    for path in (REPO_ROOT / "crates").rglob("Cargo.toml"):
        text = path.read_text(encoding="utf-8").lower()
        if "license" not in text:
            findings.append(f"{relative(path)}: no licence declared")

    pyproject = REPO_ROOT / "pyproject.toml"
    if pyproject.exists():
        text = pyproject.read_text(encoding="utf-8")
        if "GPL-3.0-only" not in text:
            findings.append("pyproject.toml: the licence is not declared as GPL-3.0-only")
    else:
        findings.append("pyproject.toml: missing")

    return findings


def check_stray_files(files: list[Path]) -> list[str]:
    """No runtime leftovers in the repository root."""
    findings: list[str] = []
    for path in REPO_ROOT.iterdir():
        if path.is_file() and path.suffix in {".jsonl", ".head"}:
            findings.append(f"{relative(path)}: stray runtime file in the repository root")
    # Machine-local binaries, walked rather than taken from ``files``: the text
    # checks only collect readable suffixes, so a .lnk never reaches this list
    # and a check placed there could not fire. A .lnk encodes an absolute path
    # to one machine's filesystem, which is personal data by construction.
    for path in sorted(REPO_ROOT.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRECTORIES for part in path.parts):
            continue
        if path.suffix.lower() in STRAY_BINARY_SUFFIXES:
            findings.append(
                f"{relative(path)}: machine-local binary ({path.suffix}) cannot be in a public repository"
            )
    return findings


CHECKS = (
    ("english", check_english),
    ("personal data", check_personal_data),
    ("secrets", check_secrets),
    ("licences", check_licences),
    ("stray files", check_stray_files),
)


def main() -> int:
    files = iter_files()
    failures = 0

    for name, check in CHECKS:
        findings = check(files)
        if findings:
            failures += len(findings)
            print(f"FAIL  {name}: {len(findings)} finding(s)")
            for finding in findings:
                print(f"      {finding}")
        else:
            print(f"ok    {name}")

    if failures:
        print(f"\n{failures} finding(s). Nothing may be published until these are fixed.")
        return 1

    print("\nAll publish checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
