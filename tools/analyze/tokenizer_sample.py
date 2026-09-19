"""Emit a deliberately awkward sample for tokenizer differential testing."""
import itertools, random, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
random.seed(7)
lines = [l.rstrip("\n") for l in itertools.islice(open(REPO / "data/corpus/docs-10k.txt"), 300)]
vocab = [l.strip() for l in open(REPO / "data/corpus/vocab.txt")]
lines += [" ".join(random.sample(vocab, 8)) for _ in range(300)]
lines += [
    "Café NAÏVE don't --- $100 3.14 e-mail's",
    "Hello,   World!!  (nested [brackets] {here})",
    "MiXeD CaSe WiTh ÀÉÎÕÜ and ñ ç ß",
    "中文 字符 mixed with english",
    "tabs\tand\nnewlines\r\nand   multiple    spaces",
    "unaffable antidisestablishmentarianism pneumonoultramicroscopicsilicovolcanoconiosis",
    "a", "", "   ", "!!!", "123 456.789 -42",
    "emoji \U0001F600 and symbols + = ^ ~ | \\ / < >",
    "hyphenated-word re-entry co-op state-of-the-art",
]
sys.stdout.write("\n".join(lines) + "\n")
