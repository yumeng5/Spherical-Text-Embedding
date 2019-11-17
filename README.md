# Spherical Text Embedding

The source code used for Spherical Text Embedding, published in NeurIPS 2019.

## Requirements

* GCC compiler (used to compile the source c file): See the [guide for installing GCC](https://gcc.gnu.org/wiki/InstallingGCC).

## Pre-trained Embeddings

We provide pre-trained ``JoSE`` embeddings on the [wikipedia dump](datasets/wiki/README.md).
* [50-d](https://drive.google.com/open?id=1bH7Jix1oQVzFxOz9ZtBa2RJZCLr6Zxvx)
* [100-d](https://drive.google.com/file/d/1hfA8BbhdnbxKejoW78lZU_voJCEfrSVH/view?usp=sharing)
* [200-d](https://drive.google.com/file/d/1qwMSFyf_6OVDxYoWywhsEhiZ3GlL041q/view?usp=sharing)
* [300-d](https://drive.google.com/file/d/13rPhPCOO1jA2ROhb4gBa8-2wsjdq-87Y/view?usp=sharing)

Unlike Euclidean embeddings such as Word2Vec and GloVe, spherical embeddings do not necessarily benefit from higher-dimensional space, so it might be a good idea to start with lower-dimensional ones first.

## Run the Code

We provide a shell script ``run.sh`` for compiling the source file and training embedding.

**Note: When preparing the training text corpus, make sure each line in the file is one document/paragraph.**

### Hyperparameters

**Note: It is recommended to use the default hyperparameters, especially the number of negative samples (``-negative``) and loss function margin (``-margin``).**

Invoke the command without arguments for a list of hyperparameters and their meanings:
```
$ ./src/jose
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
                Run more training iterations (default 10)
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

We provide a shell script ``eval_sim.sh`` for word similarity evaluation of trained spherical word embeddings on the wikipedia dump. The script will first download a zipped file of the pre-processed wikipedia dump (retrieved 2019.05; the zipped version is of ~4GB; the unzipped one is of ~13GB; for a detailed description of the dataset, see [its README file](datasets/wiki/README.md)), and then run ``JoSE`` on it. Finally, the trained embeddings are evaluated on three benchmark word similarity datasets: WordSim-353, MEN and SimLex-999.

## Document Clustering Evaluation

We provide a shell script ``eval_cluster.sh`` for document clustering evaluation of trained spherical document embeddings on the 20 Newsgroup dataset. The script will perform K-Means and Spherical K-Means clustering on the trained document embeddings.

## Document Classification Evaluation

We provide a shell script ``eval_classify.sh`` for document classification evaluation of trained spherical document embeddings on the 20 Newsgroup dataset. The script will perform KNN classification following the original 20 Newsgroup train/test split with the trained document embeddings as features.

## Citations

Please cite the following paper if you find the code helpful for your research.
```
@inproceedings{meng2019spherical,
  title={Spherical Text Embedding},
  author={Meng, Yu and Huang, Jiaxin and Wang, Guangyuan and Zhang, Chao and Zhuang, Honglei and Kaplan, Lance and Han, Jiawei},
  booktitle={Advances in neural information processing systems},
  year={2019}
}
```
