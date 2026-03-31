from pathlib import Path
import os

import numpy as np
from jax import numpy as jnp

# import roughpy as rp
import roughpy_jax as rpj
from roughpy_jax.streams import LieIncrementStream

# This should work on Debian and Ubuntu without any additional steps. On other distributions, such as Arch Linux, you
# may need to install an additional package (on Arch, the "words" package).
with open("/usr/share/dict/words", "rt") as fd:
    word_list = {word.lower() for line in fd
                 if len(word := line.strip().replace('-', '')) > 1
                 if word.isalpha() and word.isascii()}

# Some constants
ALPHABET_SIZE = 26
MAX_DEPTH = 3
RESOLUTION = 2
WORDS_LIMIT = int(os.getenv("WORDS_LIMIT", "0"))
DEFAULT_WORDS_PATHS = (
    Path("/usr/share/dict/words"),
    Path("/usr/dict/words"),
    Path("/usr/share/dict/web2"),
)
LIE_BASIS = rpj.LieBasis(width=ALPHABET_SIZE, depth=MAX_DEPTH)


def word_to_increments(word):
    rows = np.arange(len(word), dtype=np.int32)
    cols = np.fromiter((ord(letter) - 97 for letter in word), dtype=np.int32, count=len(word))
    incr_array = np.zeros((len(word), 26), dtype=np.float32)
    incr_array[rows, cols] = 1.0
    return jnp.asarray(incr_array)


def word_to_stream(word):
    timestamps = jnp.linspace(0.0, 1.0, len(word), dtype=jnp.float32)
    return LieIncrementStream.from_increments(
        timestamps=timestamps,
        data=word_to_increments(word),
        input_data_basis=None,
        resolution=RESOLUTION,
        lie_basis=LIE_BASIS,
    )


if WORDS_LIMIT > 0:
    word_list = set(sorted(word_list)[:WORDS_LIMIT])

print(f"There are {len(word_list)} words")

from collections import defaultdict
from time import time

word_streams = {word: word_to_stream(word) for word in word_list}
streams_by_length = defaultdict(list)
for word, stream in word_streams.items():
    streams_by_length[len(word)].append((word, stream))

print(f"Computed streams for {len(word_streams)} words")


def depth_key_from_log_sig(log_sig, depth):
    log_sig = log_sig.change_depth(depth)
    return str(np.asarray(log_sig.data).round(6))

# Compute and cache full-depth log signatures once, grouped by word length for
# better shape-local execution and cache locality.
start = time()
full_log_signatures = {}
for _, bucket in sorted(streams_by_length.items()):
    for word, stream in bucket:
        full_log_signatures[word] = stream.log_signature()
elapsed = time() - start
print(f"Computed full log signatures in {elapsed} seconds")


start = time()
anagrams = defaultdict(list)
for _, bucket in sorted(streams_by_length.items()):
    for word, _ in bucket:
        key = depth_key_from_log_sig(full_log_signatures[word], depth=1)
        anagrams[key].append(word)

elapsed = time() - start

print(f"Computation took {elapsed} seconds")

for key, words in anagrams.items():
    if len(words) > 6:
        print(f"{hash(key):<20}", *words)

active_words = {
    word
    for words in anagrams.values()
    if len(words) > 1
    for word in words
}

print(f"There are {len(active_words)} words with at least one anagram")

start = time()
anagrams2 = defaultdict(list)
for _, bucket in sorted(streams_by_length.items()):
    for word, _ in bucket:
        key = depth_key_from_log_sig(full_log_signatures[word], depth=2)
        if word in active_words:
            anagrams2[key].append(word)

elapsed = time() - start

print(f"Computation took {elapsed} seconds")
for key, words in anagrams2.items():
    if len(words) > 1:
        print(f"{hash(key):<20}", *words)

active_words = {
    word
    for words in anagrams2.values()
    if len(words) > 1
    for word in words
}

print(f"There are {len(active_words)} words whose level 3 signatures are necessary")


def compute_3(word_stream):
    signature = word_stream[1].signature()
    return str(np.asarray(signature.data).round(6)), word_stream[0]


start = time()
anagrams3 = defaultdict(list)
for _, bucket in sorted(streams_by_length.items()):
    for key, word in map(compute_3, bucket):
        if word in active_words:
            anagrams3[key].append(word)

elapsed = time() - start
print(f"Computation took {elapsed} seconds")

for key, words in anagrams3.items():
    if len(words) == 1:
        active_words.discard(words[0])

print(f"There are {len(active_words)} words whose level 4 signatures are necessary")
