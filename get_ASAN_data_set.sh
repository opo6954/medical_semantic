#!/bin/bash
file="largeData/data/ASAN.meta"

if [ -f "$file" ]
then
	echo "ASAN data found, no downloading..."
else
	echo "ASAN data not found, start downloading..."
	mkdir ./largeData/data
	wget -r -nH --cut-dirs=4 -np ftp://143.248.139.212/ex_storage/medical/medical_move/semantic_based_tf/exp_jh/ -P ./largeData/data/ --user=lhw --password=dnflsms
	echo "downloadExist" >> $file
fi

