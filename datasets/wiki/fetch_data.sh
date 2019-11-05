
if [ ! -e ./text.txt ] 
then
	echo ${green}===Downloading Wikipedia Dump...===${reset}
	wget --load-cookies /tmp/cookies.txt "https://drive.google.com/a/illinois.edu/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/a/illinois.edu/uc?export=download&id=1fT1GxBMXEItf2NtNMjdXhA61HPecPO98' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1fT1GxBMXEItf2NtNMjdXhA61HPecPO98" -O wiki_dump.zip && rm -rf /tmp/cookies.txt
	echo ${green}===Unzipping Wikipedia Dump...===${reset}
	unzip wiki_dump.zip && rm wiki_dump.zip
fi
