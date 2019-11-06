# dataset directory
dataset=20news

# text file name; one document per line
text_file=text.txt

# document embedding output file name
doc_emb=jose_d.txt

# word embedding dimension
word_dim=100

# local context window size
window_size=10

# minimum word count in corpus; words that appear less than this threshold will be discarded
min_count=5

# number of iterations to run on the corpus
iter=20

# number of threads to be run in parallel
threads=10

cd ./src
make jose
cd ..

start=$SECONDS

./src/jose -train ./datasets/${dataset}/${text_file} -doc-output ./datasets/${dataset}/${doc_emb} \
	-size ${word_dim} -alpha 0.04 -margin 0.15 -window ${window_size} -negative 2 -sample 1e-3 \
	-min-count ${min_count} -iter ${iter} -threads ${threads} 

duration=$(( SECONDS - start ))
printf '\nRunning time is %s seconds.\n' "$duration"

emb_file=${doc_emb}

# evaluate document classification with KNN
# 20 Newsgroup has 11314 training documents; the remaining ones are for testing
train_num=11314
python classify.py --dataset ${dataset} --emb_file ${emb_file} --train_num ${train_num}
