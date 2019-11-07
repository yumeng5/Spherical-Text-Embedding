# Wikipedia Dump Dataset

We provide a script ``fetch_data.sh`` to download the pre-processed wikipedia dump. Alternatively, you can manually download the file from [this Google Drive link](https://drive.google.com/open?id=1fT1GxBMXEItf2NtNMjdXhA61HPecPO98) and upzip it to obtain the text corpus.  
**Note: If you have run ``eval_sim.sh`` from the root directory, the wikipedia dump should have already been automatically downloaded under this directory, and you will not need to run ``fetch_data.sh`` or manually download the dataset.**

## Dataset Description

The dataset is retrieved from the [wikipedia database dump](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2) on 2019.05. The corpus is pre-processed using [Stanford CoreNLP tookit](https://stanfordnlp.github.io/CoreNLP/index.html#download) for sentence tokenization. Each line of the text file contains one wikipedia paragraph. The zipped file (to be downloaded) is of ~4GB. After unzipping the file, the corpus size is ~13GB.
