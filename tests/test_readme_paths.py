import re
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
README_PATHS = [ROOT_DIR / "README.md", ROOT_DIR / "app" / "readme.md"]
LOCAL_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def normalize_local_target(raw_target: str) -> Path | None:
    target = raw_target.split("#", 1)[0].strip()
    if not target:
        return None
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target):
        return None
    if target.startswith(("mailto:", "http://", "https://")):
        return None
    target = target.split(":", 1)[0] if re.match(r"^[A-Za-z]:/", target) else target
    candidate = Path(target)
    if candidate.is_absolute():
        return candidate
    return (ROOT_DIR / candidate).resolve()


def test_readme_local_links_exist() -> None:
    missing = []
    for readme_path in README_PATHS:
        assert readme_path.exists(), f"Missing README file: {readme_path}"
        text = readme_path.read_text(encoding="utf-8", errors="ignore")
        for match in LOCAL_LINK_RE.finditer(text):
            target = normalize_local_target(match.group(1))
            if target is not None and not target.exists():
                missing.append(f"{readme_path}: {match.group(1)}")

    assert not missing
