# dataset directory (need to be under ./datasets)
dataset=DIRECTORY_TO_CORPUS

# text file name; one document per line
text_file=text.txt

# word embedding output file name
out_file=jose.txt

# word embedding dimension
word_dim=100

# local context window size
window_size=10

# minimum word count in corpus; words that appear less than this threshold will be discarded
min_count=5

# number of iterations to run on the corpus
iter=10

# number of threads to be run in parallel
threads=20

cd ./src
make jose
cd ..

start=$SECONDS

./src/jose -train ./datasets/${dataset}/${text_file} -word-output ./datasets/${dataset}/${out_file} \
	-size ${word_dim} -alpha 0.04 -margin 0.15 -window ${window_size} -negative 2 -sample 1e-3 \
	-min-count ${min_count} -iter ${iter} -threads ${threads} 

duration=$(( SECONDS - start ))
printf '\nRunning time %s\n' "$duration"
