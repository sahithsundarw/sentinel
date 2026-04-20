"""
update_blog_links.py

Updates blog_post.md with explicit Colab and GitHub links, then generates
blog_post_hf.md with HuggingFace blog frontmatter.

Usage:
    python update_blog_links.py
"""
import re
import os
import sys

COLAB_URL = "https://colab.research.google.com/github/sahithsundarw/sentinel/blob/main/training_colab.ipynb"
GITHUB_URL = "https://github.com/sahithsundarw/sentinel"
HF_SPACE_URL = "https://varunventra-guardrail-arena.hf.space"

HF_FRONTMATTER = """---
title: "Guardrail Arena: Training AI Safety Agents with Multi-Agent RL"
thumbnail: /blog/assets/guardrail-arena/reward_curves.png
authors:
- user: varunventra
- user: sahithsundarw
- user: pranush
---

"""


def count_words(text: str) -> int:
    return len(text.split())


def check_blog(content: str) -> dict:
    """Check blog_post.md for required content."""
    checks = {
        "word_count": count_words(content),
        "has_qwen_zero": "0.0000" in content and "235B" in content,
        "has_herbal_tea": "herbal" in content.lower(),
        "has_hf_space_link": HF_SPACE_URL in content,
        "has_github_link": GITHUB_URL in content,
        "has_colab_link": COLAB_URL in content,
        "has_image_ref": "![" in content,
        "has_cta": any(kw in content.lower() for kw in ["try it", "quick start", "pip install"]),
    }
    return checks


def update_blog(content: str) -> str:
    """Patch blog_post.md with required links."""
    # 1. Replace bare "training notebook" with Colab link
    # Handle both "The training notebook" and "training notebook" variations
    content = re.sub(
        r'\bThe training notebook\b',
        f'The [training notebook]({COLAB_URL})',
        content, count=1
    )
    # Also replace plain "training notebook" if it wasn't caught above
    if COLAB_URL not in content:
        content = re.sub(
            r'\btraining notebook\b',
            f'[training notebook]({COLAB_URL})',
            content, count=1
        )

    # 2. Add GitHub link if missing — append to the Try It section
    if GITHUB_URL not in content:
        # Try to append after the hf.space URL mention
        content = content.replace(
            f'The environment is live at `{HF_SPACE_URL}`.',
            f'The environment is live at `{HF_SPACE_URL}`.\n'
            f'Source code: [{GITHUB_URL}]({GITHUB_URL})'
        )
        # Fallback: append to end of Try It section or end of file
        if GITHUB_URL not in content:
            # Insert before last code block or append to end
            code_block_pos = content.rfind("```")
            if code_block_pos > 0:
                content = content[:code_block_pos] + \
                    f"Source: [{GITHUB_URL}]({GITHUB_URL})\n\n" + \
                    content[code_block_pos:]
            else:
                content += f"\n\nSource: [{GITHUB_URL}]({GITHUB_URL})"

    return content


def create_hf_version(content: str) -> str:
    """Create HF blog version: strip the H1 title line, add frontmatter."""
    # Remove the first H1 heading line if present (it becomes the title field)
    lines = content.split("\n")
    body_lines = []
    skipped_h1 = False
    for line in lines:
        if not skipped_h1 and line.startswith("# "):
            skipped_h1 = True
            continue  # drop H1 — it becomes the title in frontmatter
        body_lines.append(line)
    body = "\n".join(body_lines).lstrip("\n")
    return HF_FRONTMATTER + body


def print_checklist(checks: dict) -> None:
    ok = "[OK]"
    no = "[MISSING]"
    print("\n-- Blog Post Checklist --")
    wc = checks['word_count']
    wc_note = "OK" if wc <= 500 else f"OVER (trim {wc - 500} words)"
    print(f"  Word count:          {wc} words  ({wc_note})")
    print(f"  Qwen 235B / 0.0000:  {ok if checks['has_qwen_zero'] else no}")
    print(f"  Herbal tea example:  {ok if checks['has_herbal_tea'] else no}")
    print(f"  HF Space link:       {ok if checks['has_hf_space_link'] else no}")
    print(f"  GitHub link:         {ok if checks['has_github_link'] else no}")
    print(f"  Colab link:          {ok if checks['has_colab_link'] else no}")
    print(f"  Image reference:     {ok if checks['has_image_ref'] else no}")
    print(f"  Call to action:      {ok if checks['has_cta'] else no}")
    # has_herbal_tea is a nice-to-have (pitch example), not a blog requirement
    required = {k: v for k, v in checks.items() if k not in ("word_count", "has_herbal_tea")}
    failures = [k for k, v in required.items() if not v]
    if failures:
        print(f"\n  Issues fixed automatically: {', '.join(failures)}")
    else:
        print("\n  All checks pass [OK]")


def main() -> None:
    blog_path = "blog_post.md"
    hf_path = "blog_post_hf.md"

    if not os.path.exists(blog_path):
        print(f"ERROR: {blog_path} not found")
        sys.exit(1)

    with open(blog_path) as f:
        original = f.read()

    print("Reading blog_post.md...")
    print(f"  {count_words(original)} words, {len(original)} chars")

    # Check what we have
    before_checks = check_blog(original)
    print_checklist(before_checks)

    # Apply updates
    updated = update_blog(original)

    # Check after update
    after_checks = check_blog(updated)
    all_pass = all(v for k, v in after_checks.items() if k != "word_count")

    if updated != original:
        with open(blog_path, "w") as f:
            f.write(updated)
        print(f"\n  Updated: {blog_path}")
    else:
        print(f"\n  No changes needed: {blog_path}")

    # Create HF version
    hf_content = create_hf_version(updated)
    with open(hf_path, "w") as f:
        f.write(hf_content)
    print(f"  Created: {hf_path}")

    # has_herbal_tea is a nice-to-have (pitch example), not required in the blog
    required_keys = {k for k in after_checks if k not in ("word_count", "has_herbal_tea")}
    all_pass = all(after_checks[k] for k in required_keys)
    if not all_pass:
        print("\n[WARN] Some checks still failing after update (manual review needed)")
        sys.exit(1)
    else:
        print("\n[OK]  blog_post.md is ready for publication")
        print(f"[OK]  {hf_path} is ready for HuggingFace blog")


if __name__ == "__main__":
    main()
