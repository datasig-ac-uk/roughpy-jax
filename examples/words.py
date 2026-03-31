from pathlib import Path

# import numpy as np
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
DEFAULT_WORDS_PATHS = (
    Path("/usr/share/dict/words"),
    Path("/usr/dict/words"),
    Path("/usr/share/dict/web2"),
)
LIE_BASIS = rpj.LieBasis(width=ALPHABET_SIZE, depth=MAX_DEPTH)


def word_to_stream(word):
    incr_array = jnp.zeros((len(word), 26), dtype=jnp.float32)
    for i, letter in enumerate(word):
        assert 97 <= ord(letter) <= 122, f"{letter} is not allowed"
        incr_array = incr_array.at[i, ord(letter) - 97].set(1)

    return LieIncrementStream.from_increments(
        timestamps=jnp.linspace(0.0, 1.0, len(word), dtype=jnp.float32),
        data=incr_array, 
        input_data_basis=None,
        resolution=RESOLUTION,
        lie_basis=LIE_BASIS
        )


print(f"There are {len(word_list)} words")

word_streams = {word: word_to_stream(word) for word in word_list}

print(f"Computed streams for {len(word_streams)} words")

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from time import time


def compute(word_stream, *, depth):
    """"Helper function for to get (key, word) results"""
    return str(word_stream[1].log_signature(depth=depth)), word_stream[0]


start = time()
anagrams = defaultdict(list)
with ThreadPoolExecutor(max_workers=8) as pool:
    for key, word in pool.map(partial(compute, depth=1), word_streams.items()):
        anagrams[key].append(word)

elapsed = time() - start

print(f"Computation took {elapsed} seconds")

for key, words in anagrams.items():
    if len(words) == 1:
        word_streams.pop(words[0])
    if len(words) > 6:
        print(f"{key:<40}", *words)




print(f"There are {len(word_streams)} words with at least one anagram")

start = time()
anagrams2 = defaultdict(list)
with ThreadPoolExecutor() as pool:
    for key, word in pool.map(partial(compute, depth=2), word_streams.items()):
        anagrams2[key].append(word)

elapsed = time() - start

print(f"Computation took {elapsed} seconds")
for key, words in anagrams2.items():
    if len(words) == 1:
        word_streams.pop(words[0])
    else:
        print(f"{key:<40}", *words)

print(f"There are {len(word_streams)} words whose level 3 signatures are necessary")


def compute_3(word_stream):
    return str(word_stream[1].signature()), word_stream[0]


start = time()
anagrams3 = defaultdict(list)
with ThreadPoolExecutor() as pool:
    for key, word in pool.map(compute_3, word_streams.items()):
        anagrams3[key].append(word)

elapsed = time() - start
print(f"Computation took {elapsed} seconds")

for key, words in anagrams3.items():
    if len(words) == 1:
        word_streams.pop(words[0])

print(f"There are {len(word_streams)} words whose level 4 signatures are necessary")
