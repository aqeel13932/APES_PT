tst="_Test.avi"
title=" Test"
for j in `seq 1 3`;
do
	cd $j/VID
	for t in $(ls);
	do
		t=${t%_*}
		echo "$t$tst"
		python ../../youtubeUploader.py --file "$t$tst" --title "$t$title" --logging_level DEBUG --description E$j
	done
	cd ../../
done
