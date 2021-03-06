#!/bin/bash
file="largeData/data_png/ASAN.meta"

if [ -f "$file" ]
then
	echo "ASAN data_png found, no downloading..."
else
	echo "ASAN data_png not found, start downloading..."
	mkdir ./largeData/data_png
	wget -r -nH --cut-dirs=4 -np ftp://143.248.139.212/ex_storage/medical/medical_move/semantic_based_tf/wholeData_bbox/ -P ./largeData/data_png/ --user=ID --password=ID
	wget -r -nH --cut-dirs=4 -np ftp://143.248.139.212/ex_storage/medical/medical_move/semantic_based_tf/wholeData_mask/ -P ./largeData/data_png/ --user=ID --password=ID
	wget -r -nH --cut-dirs=4 -np ftp://143.248.139.212/ex_storage/medical/medical_move/semantic_based_tf/wholeData_origin/ -P ./largeData/data_png/ --user=ID --password=ID
	echo "downloadExist" >> $file
fi

