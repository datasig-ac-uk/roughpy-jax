# Examples

This folder contains examples of how to use the `PiecewiseAbelianStream` and 
`LieIncrementStream` with the words example found in the original RoughPy. In 
short, it goes through a long wordlist (found at `/usr/share/dict/words' on most
Linux systems), creates a stream for each and then calculates the log-signature 
at various depths. 

Note that a couple of environment variables have been added to ease the 
configurability:
- `WORDS_LIMIT` - changes how many words from the list are selected. Defaults to
the whole list if not set. 
- `WORDS_STREAM_KIND` - sets which stream type to use, with valid values being 
`piecewise_abelian` or `lie_increment`. Defaults to lie_increment.

The example can therefore be run with a simple:

```
python examples/words.py
```

But for a shorter example use 

```
WORDS_LIMIT=1000 WORDS_STREAM_KIND=piecewise_abelian python examples/words.py
```

The piecewise Abelian stream is faster to create but slower to query, while the 
Lie increment stream is slower to create but faster to query - due to the 
creation of the dyadic cache. 