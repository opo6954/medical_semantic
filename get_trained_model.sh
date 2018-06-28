#!/bin/bash
file="largeData/model/trainedModel.meta"

if [ -f "$file" ]
then
	echo "trained model found, no downloading..."
else
	echo "trained model not found, start downloading..."
	wget -r -nH --cut-dirs=7 -np ftp://143.248.139.212/ex_storage/medical/medical_move/semantic_based_source/modelStore/ASAN/180627/modelTrainedPath/ -P ./largeData/model/ --user=lhw --password=dnflsms
	echo "downloadExist" >> $file
fi

