# Language-Model (Text-Generation)
This repository contains my implementations of Language Model for Text Generation which are extended from [My CS224d Assignment Solutions](https://github.com/Krishnkant-Swarnkar/CS224d-Assignment-Solutions/blob/master/Assignment2/q3_RNNLM.py)

## How to Run
Simply run 
~~~~{.python}
python generate_text.py
~~~~
give it some starter text and enjoy text generaions from the system :-)

## Models
* 1 Layer LSTM, word_embedding_size = 50, hidden_state_size = 100 , Test perplexity on PTB: 147.298156738

Some examples as produced by the system:
~~~~
STARTER TEXT> microsoft shares
GENERATED TEXT> microsoft shares produced N million francs in three two months of these small investors.

STARTER TEXT> ozone is
GENERATED TEXT> ozone is still changed from the space.

STARTER TEXT> microsoft
GENERATED TEXT> microsoft said taiwan mci communications corp. closed at $ N a share .

STARTER TEXT> share
GENERATED TEXT> share of aetna may have backed up in the stock market <unk>.
~~~~
