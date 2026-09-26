"""Create and seed a deterministic synthetic EpisodicFS work history."""

import argparse
import os
import subprocess
import tempfile
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from PIL import Image, ImageDraw


EPISODE_STARTS = {
    "q3_planning": datetime(2025, 7, 8, 9, 0, tzinfo=timezone.utc),
    "pipeline_debug": datetime(2025, 7, 10, 14, 0, tzinfo=timezone.utc),
}


def _write(path: Path, content: str, timestamp: datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    epoch = timestamp.timestamp()
    os.utime(path, (epoch, epoch))


def _write_pdf(path: Path, title: str, body: str, timestamp: datetime) -> None:
    """Write a minimal text PDF without adding another fixture dependency."""
    stream = f"BT /F1 12 Tf 72 720 Td ({title}) Tj 0 -24 Td ({body}) Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(stream.encode('latin-1'))} >>\nstream\n{stream}\nendstream".encode("latin-1"),
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = []
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii"))
    pdf.extend("".join(f"{offset:010d} 00000 n \n" for offset in offsets).encode("ascii"))
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pdf)
    epoch = timestamp.timestamp()
    os.utime(path, (epoch, epoch))


def _commit_code_history(code_dir: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=code_dir, check=True)
    subprocess.run(["git", "config", "user.email", "fixtures@episodicfs.local"], cwd=code_dir, check=True)
    subprocess.run(["git", "config", "user.name", "EpisodicFS Fixtures"], cwd=code_dir, check=True)
    subprocess.run(["git", "add", "."], cwd=code_dir, check=True)
    subprocess.run(["git", "commit", "-qm", "Add pipeline baseline"], cwd=code_dir, check=True)
    _write(
        code_dir / "pipeline.py",
        """from pathlib import Path\n\ndef load_events(path: str):\n    return Path(path).read_text().splitlines()\n\ndef normalize_event(event: str) -> str:\n    return event.strip().lower()\n""",
        EPISODE_STARTS["pipeline_debug"],
    )
    subprocess.run(["git", "add", "."], cwd=code_dir, check=True)
    subprocess.run(["git", "commit", "-qm", "Debug event pipeline normalization"], cwd=code_dir, check=True)


def generate_fixtures(base_dir: Optional[str] = None) -> Path:
    """Create a synthetic vault and return its root path.

    With no path supplied, fixtures are written to a new temporary directory.
    """
    root = Path(base_dir) if base_dir else Path(tempfile.mkdtemp(prefix="episodicfs_seed_"))
    code_dir = root / "code" / "event_pipeline"
    docs_dir = root / "docs"
    image_dir = root / "images"
    audio_dir = root / "audio"

    q3 = EPISODE_STARTS["q3_planning"]
    debug = EPISODE_STARTS["pipeline_debug"]
    _write(code_dir / "README.md", "# Event Pipeline\nQ3 planning data pipeline and metrics service.\n", q3)
    _write(code_dir / "pipeline.py", "def load_events(path):\n    return open(path).read().splitlines()\n", q3)
    _write(code_dir / "tests" / "test_pipeline.py", "def test_load_events():\n    assert True\n", q3.replace(hour=10))

    _write(docs_dir / "q3_planning_notes.md", "# Q3 planning session\nPrioritize pipeline reliability, event latency, and dashboard milestones.\n", q3.replace(minute=5))
    _write(docs_dir / "roadmap.txt", "Q3 roadmap: data pipeline reliability, analytics dashboard, and hiring milestones.\n", q3.replace(minute=18))
    _write(docs_dir / "planning_transcript.txt", "Meeting transcript: the team reviewed Q3 pipeline capacity and dashboard launch dates.\n", q3.replace(minute=27))
    _write_pdf(docs_dir / "invoice_analytics.pdf", "Invoice #Q3-104", "Analytics dashboard planning workshop - 2400 USD", q3.replace(minute=35))

    _write(code_dir / "debug_notes.md", "Pipeline debug: retry queue duplicates events after a timeout. Inspect normalization and batch offsets.\n", debug)
    _write(code_dir / "fixtures.json", '{"event": "purchase", "retry": true}\n', debug.replace(minute=8))
    _write(docs_dir / "pipeline_incident.txt", "Incident transcript: event ingestion latency rose after the retry queue change.\n", debug.replace(minute=16))
    _write_pdf(docs_dir / "invoice_cloud_debug.pdf", "Invoice #OPS-221", "Cloud pipeline incident response and observability review - 1800 USD", debug.replace(minute=24))

    colors = [(32, 96, 144), (144, 80, 48)]
    for index, color in enumerate(colors, start=1):
        image = Image.new("RGB", (160, 100), color)
        ImageDraw.Draw(image).text((12, 42), f"Episode {index}", fill="white")
        image_path = image_dir / f"episode_{index}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(image_path)
        epoch = (q3 if index == 1 else debug).timestamp()
        os.utime(image_path, (epoch, epoch))

    audio_path = audio_dir / "planning_voice_note.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(audio_path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b"\x00\x00" * 8000)
    os.utime(audio_path, (q3.replace(minute=42).timestamp(),) * 2)

    _commit_code_history(code_dir)
    print(f"Created synthetic fixture vault at {root}")
    return root


def run_seed(base_dir: Optional[str] = None, time_window_seconds: int = 30 * 60) -> Dict[str, object]:
    """Generate fixtures and index them through the normal EpisodicFS pipeline."""
    root = generate_fixtures(base_dir)
    from src.db import setup_and_ingest_lancedb

    database = setup_and_ingest_lancedb(str(root), time_window_seconds=time_window_seconds)
    return {"base_dir": str(root), "db": database}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_dir", default=None, help="Fixture root; defaults to a temporary directory.")
    parser.add_argument("--seed", action="store_true", help="Also index fixtures into LanceDB and the episodic graph.")
    args = parser.parse_args()
    result = run_seed(args.base_dir) if args.seed else {"base_dir": str(generate_fixtures(args.base_dir))}
    print(f"Fixture base directory: {result['base_dir']}")
