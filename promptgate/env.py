"""Read API keys from a .env file in the project folder.

A .env file holds lines like ANTHROPIC_API_KEY=sk-ant-... and stays on this
machine (it is in .gitignore). A variable that is already set in the real
environment always wins, so the file never replaces a key you set in Windows.

Only the promptgate command loads this file. Tests call cli.main() directly,
so they never read your real keys.
"""

import os
from pathlib import Path

from promptgate.config import PROJECT_ROOT

DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


def parse_env_line(line: str) -> tuple[str, str] | None:
    """Turn 'NAME=value' into ("NAME", "value"). Returns None for blank lines and # comments.

    Quotes around the value are removed, so NAME="value" also works.
    Everything after the first "=" is the value, so values may contain "=".
    """
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        return None
    name, value = line.split("=", 1)
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return name.strip(), value


def load_env_file(path: Path = DEFAULT_ENV_FILE) -> list[str]:
    """Copy each NAME=value from the file into the environment, unless NAME is already set.

    Empty values are skipped, so a blank line like ANTHROPIC_API_KEY= counts
    as "not set" and promptgate can say which key is missing.
    Returns the names that were set. A missing file sets nothing.
    """
    if not path.exists():
        return []
    loaded = []
    # utf-8-sig also reads files saved by Notepad, which adds an invisible marker at the start.
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        parsed = parse_env_line(line)
        if parsed is None:
            continue
        name, value = parsed
        if value and not os.environ.get(name):
            os.environ[name] = value
            loaded.append(name)
    return loaded
