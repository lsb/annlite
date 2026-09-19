"""Token ids for lines on stdin, matching examples/tokenize_dump.rs output."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
from embed.tokenization import WordPieceTokenizer  # noqa: E402

t = WordPieceTokenizer(REPO / "models/tokenizers/bert-base-uncased-vocab.txt")
for line in sys.stdin:
    print(",".join(str(i) for i in t.encode(line.rstrip("\n"), 512)))
