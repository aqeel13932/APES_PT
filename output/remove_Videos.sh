for i in `seq 1 186`;
do
	cd $i
	ls
	rm -rf VID
	cd ..
done
