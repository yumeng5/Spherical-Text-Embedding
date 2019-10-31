# dataset directory
dataset=wiki

# text file name; one document per line
text_file=text.txt

# word embedding output file name
out_file=jose.txt

# word embedding dimension
word_dim=100

# local context window size
window_size=10

# minimum word count in corpus; words that appear less than this threshold will be discarded
min_count=100

# number of iterations to run on the corpus
iter=10

# number of threads to be run in parallel
threads=20

green=`tput setaf 2`
reset=`tput sgr0`

make jose

start=$SECONDS

if [ ! -e ./datasets/${dataset}/text.txt ] 
then
	cd ./datasets/${dataset}/
	echo ${green}===Downloading Wikipedia Dump...===${reset}
	wget --load-cookies /tmp/cookies.txt "https://drive.google.com/a/illinois.edu/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/a/illinois.edu/uc?export=download&id=1fT1GxBMXEItf2NtNMjdXhA61HPecPO98' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1fT1GxBMXEItf2NtNMjdXhA61HPecPO98" -O wiki_dump.zip && rm -rf /tmp/cookies.txt
	echo ${green}===Unzipping Wikipedia Dump...===${reset}
	unzip wiki_dump.zip
	cd ../../
fi

./jose -train ./datasets/${dataset}/${text_file} -word-output ./datasets/${dataset}/${out_file} \
	-size ${word_dim} -alpha 0.04 -margin 0.15 -window ${window_size} -negative 2 -sample 1e-3 \
	-min-count ${min_count} -iter ${iter} -threads ${threads} 

duration=$(( SECONDS - start ))
printf '\nRunning time is %s seconds.\n' "$duration"

emb_file=${out_file}
python sim.py --dataset ${dataset} --emb_file ${emb_file}
