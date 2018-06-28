#!/bin/bash
file="largeData/cifar10/cifar10.meta"

if [ -f "$file" ]
then
	echo "cifar10 data found, no downloading..."
else
	echo "cifar10 data not found, start downloading"
	wget -r -nH --cut-dirs=4 -np ftp://143.248.139.212/ex_storage/medical/medical_move/semantic_based_tf/exp_cifar10/ -P ./largeData/cifar10/ --user=lhw --password=dnflsms
	echo "downloadExist" >> $file
fi
