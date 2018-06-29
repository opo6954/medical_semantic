#!/bin/bash
file="largeData/modelAlexnet/bvlc_alexnet.npy"

if [ -f "$file" ]
then
	echo "$file found... No downloading..."
else
	echo "$file not found... Start downloading..."
	mkdir ./largeData/modelAlexnet
	wget ftp://143.248.139.212/ex_storage/medical/medical_move/semantic_based_tf/Alexnet/bvlc_alexnet.npy -O ./largeData/modelAlexnet/bvlc_alexnet.npy --user=lhw --password=dnflsms
fi
