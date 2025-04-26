import json
import hashlib
from textwrap import wrap
from pathlib import Path
import argparse
import sys

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import configuration
from config import (
    INPUT_DATA_FILE,
    RAG_CHUNKS_FILE,
    DEFAULT_CHUNK_SIZE,
    ensure_directories
)

def generate_chunk_id(entry, i):
    base = entry.get("doi") or entry.get("coreId") or f"entry_{i}"
    return hashlib.sha256(f"{base}_{i}".encode()).hexdigest()

def preprocess_entry(entry, chunk_size=512):
    raw = f"{entry.get('title','')}\n\n{entry.get('fullText','')}".strip()
    text = " ".join(raw.split())
    pieces = wrap(text, chunk_size)
    meta = {
        "title":    entry.get("title"),
        "authors":  entry.get("authors"),
        "year":     entry.get("year"),
        "coreId":   entry.get("coreId"),
        "doi":      entry.get("doi"),
        "source":   entry.get("publisher", "Unknown")
    }
    return [
        {"id": generate_chunk_id(entry, i), "text": chunk, "meta": meta}
        for i, chunk in enumerate(pieces)
    ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=str(INPUT_DATA_FILE), help="Path to input NDJSON file")
    parser.add_argument("--output", default=str(RAG_CHUNKS_FILE), help="Output JSON file path")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE, help="Max characters per chunk")
    args = parser.parse_args()

    # Ensure directories exist
    ensure_directories()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        print(f"❌ Cannot find {in_path}")
        return

    all_chunks = []
    with in_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("fullText"):
                    all_chunks.extend(preprocess_entry(entry, chunk_size=args.chunk_size))
            except json.JSONDecodeError as e:
                print(f"⚠️ Line {line_num}: Skipping malformed JSON ({e})")

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"✅ Wrote {len(all_chunks)} chunks to {out_path}")

if __name__ == "__main__":
    main()
