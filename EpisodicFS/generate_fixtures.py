"""Generate small, deterministic media fixtures for a local EpisodicFS demo."""

import argparse
import wave
from pathlib import Path

from PIL import Image, ImageDraw


def generate_fixtures(base_dir="episodic_vault"):
    root = Path(base_dir)
    image_dir = root / "images"
    docs_dir = root / "docs"
    audio_dir = root / "audio"
    for directory in (image_dir, docs_dir, audio_dir):
        directory.mkdir(parents=True, exist_ok=True)

    colors = [(32, 96, 144), (144, 80, 48), (64, 128, 80)]
    for index, color in enumerate(colors, start=1):
        image = Image.new("RGB", (160, 100), color)
        ImageDraw.Draw(image).text((12, 42), f"EpisodicFS {index}", fill="white")
        image.save(image_dir / f"sample_{index}.png")

    (docs_dir / "meeting_notes.txt").write_text(
        "Server upgrade discussion with Rahul. Follow up on deployment timing.\n",
        encoding="utf-8",
    )
    (docs_dir / "restaurant_receipt.md").write_text(
        "# Restaurant receipt\nDinner after the server upgrade meeting.\n",
        encoding="utf-8",
    )

    with wave.open(str(audio_dir / "sample_voice.wav"), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b"\x00\x00" * 8000)

    print(f"Created fixture vault at {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_dir", default="episodic_vault")
    generate_fixtures(parser.parse_args().base_dir)