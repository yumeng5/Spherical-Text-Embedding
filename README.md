# Spherical Text Embedding

The source code used for Spherical Text Embedding, published in NeurIPS 2019.

## Requirements

* GCC compiler (used to compile the source c file): See the [guide for installing GCC](https://gcc.gnu.org/wiki/InstallingGCC).

## Run the Code

We provide a shell script ``run.sh`` for compiling the source file and training embedding.

### Hyperparameters

Invoke the command without arguments for a list of hyperparameters and their meanings:
```
$ ./jose
Parameters:
        -train <file> (mandatory argument)
                Use text data from <file> to train the model
        -word-output <file>
                Use <file> to save the resulting word vectors
        -context-output <file>
                Use <file> to save the resulting word context vectors
        -doc-output <file>
                Use <file> to save the resulting document vectors
        -size <int>
                Set size of word vectors; default is 100
        -window <int>
                Set max skip length between words; default is 5
        -sample <float>
                Set threshold for occurrence of words. Those that appear with higher frequency in the
                training data will be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)
        -negative <int>
                Number of negative examples; default is 2
        -threads <int>
                Use <int> threads (default 20)
        -margin <float>
                Margin used in loss function to separate positive samples from negative samples
        -iter <int>
                Run more training iterations (default 5)
        -min-count <int>
                This will discard words that appear less than <int> times; default is 5
        -alpha <float>
                Set the starting learning rate; default is 0.04
        -debug <int>
                Set the debug mode (default = 2 = more info during training)
        -save-vocab <file>
                The vocabulary will be saved to <file>
        -read-vocab <file>
                The vocabulary will be read from <file>, not constructed from the training data
        -load-emb <file>
                The pretrained embeddings will be read from <file>

Examples:
./jose -train text.txt -word-output jose.txt -size 100 -margin 0.15 -window 5 -sample 1e-3 -negative 2 -iter 10

```

## Word Similarity Evaluation

