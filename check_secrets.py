"""
check_secrets.py

Scans the current working tree (not git history) for accidentally committed secrets.
Does NOT scan git history (that requires git-filter-repo and is destructive).

Usage:
    python check_secrets.py

Exit code 0 if clean, exit code 1 if secrets found.
"""
import os
import re
import sys

# Patterns to scan for
SECRET_PATTERNS = [
    (re.compile(r'hf_[A-Za-z]{34,}'), "HuggingFace token"),
    (re.compile(r'sk-[A-Za-z0-9]{32,}'), "OpenAI API key"),
    (re.compile(r'OPENAI_API_KEY\s*=\s*[^\s$\'"\n][^\n]*'), "OPENAI_API_KEY assignment"),
    (re.compile(r'HF_TOKEN\s*=\s*[^\s$\'"\n][^\n]*'), "HF_TOKEN assignment"),
    (re.compile(r'ANTHROPIC_API_KEY\s*=\s*[^\s$\'"\n][^\n]*'), "Anthropic API key"),
]

# Value sub-strings that indicate placeholder / example text — skip these hits
_PLACEHOLDER_TOKENS = (
    "your_", "your-", "_your", "placeholder", "xxx", "<token", "<key",
    "os.environ", "os.getenv", "getenv(", "environ.get(", "environ[",
    "example", "replace_", "INSERT", "REPLACE", "<REVOKED>",
)


def _is_placeholder(line: str) -> bool:
    """Return True if the matched line looks like documentation/example, not a real credential."""
    lower = line.lower()
    return any(tok.lower() in lower for tok in _PLACEHOLDER_TOKENS)

# File extensions to scan
SCAN_EXTENSIONS = {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".sh", ".bat", ".env", ".toml"}

# Directories and files to skip
SKIP_DIRS = {".git", "venv", "venv311", "venv310", ".venv", "__pycache__", "node_modules", ".pytest_cache"}
SKIP_FILES = {"check_secrets.py"}  # self-exclusion


def should_skip(path: str) -> bool:
    parts = path.replace("\\", "/").split("/")
    for part in parts:
        if part in SKIP_DIRS:
            return True
    return False


def scan_file(filepath: str) -> list[tuple[int, str, str]]:
    """Return list of (line_number, pattern_name, matched_text) for any hits."""
    hits = []
    try:
        with open(filepath, encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, 1):
                for pattern, name in SECRET_PATTERNS:
                    m = pattern.search(line)
                    if m:
                        if _is_placeholder(line):
                            continue  # skip documentation / example lines
                        # Redact most of the match for display
                        matched = m.group(0)
                        display = matched[:12] + "..." + matched[-4:] if len(matched) > 20 else matched[:8] + "..."
                        hits.append((lineno, name, display))
    except (PermissionError, IsADirectoryError):
        pass
    return hits


def main() -> None:
    root = os.getcwd()
    print(f"Scanning working tree for secrets: {root}")
    print(f"Patterns: {len(SECRET_PATTERNS)}")
    print()

    total_scanned = 0
    findings: list[tuple[str, int, str, str]] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skip directories in-place
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        rel_dir = os.path.relpath(dirpath, root)
        if should_skip(rel_dir):
            continue

        for fname in filenames:
            if fname in SKIP_FILES:
                continue
            _, ext = os.path.splitext(fname)
            if ext.lower() not in SCAN_EXTENSIONS:
                continue

            fpath = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(fpath, root)
            if should_skip(rel_path):
                continue

            total_scanned += 1
            hits = scan_file(fpath)
            for lineno, name, display in hits:
                findings.append((rel_path, lineno, name, display))

    print(f"Files scanned: {total_scanned}")
    print()

    if findings:
        print(f"[FAIL]  {len(findings)} potential secret(s) found:\n")
        for fpath, lineno, name, display in findings:
            print(f"  {fpath}:{lineno}  [{name}]  {display}")
        print()
        print("Action required:")
        print("  1. Remove the secret from the file")
        print("  2. Rotate/revoke the token at huggingface.co/settings/tokens (or OpenAI/Anthropic dashboard)")
        print("  3. The git history may still contain it -- use git-filter-repo to scrub history")
        print("  4. Do NOT force-push to HF Space -- use the orphan branch workflow in README")
        sys.exit(1)
    else:
        print(f"[OK]  No secrets found in working tree ({total_scanned} files scanned)")
        print()
        print("Note: This script scans the WORKING TREE only.")
        print("The git history may still contain exposed tokens from older commits.")
        print("Always use the orphan branch workflow when pushing to HuggingFace Space.")
        sys.exit(0)


if __name__ == "__main__":
    main()
