from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return cleaned[:80] or "retail-research"


def save_report(repo_path: Path, query: str, report_text: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{slugify(query)}.txt"
    target = repo_path / filename
    target.write_text(report_text, encoding="utf-8")
    return target


def list_saved_reports(repo_path: Path) -> list[dict[str, str]]:
    reports = []
    for path in sorted(repo_path.glob("*.txt"), reverse=True):
        text = path.read_text(encoding="utf-8")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        title = lines[0].replace("# ", "") if lines else path.stem
        preview = lines[1] if len(lines) > 1 else "Saved retail research report."
        reports.append(
            {
                "title": title,
                "filename": path.name,
                "preview": preview[:180],
            }
        )
    return reports
