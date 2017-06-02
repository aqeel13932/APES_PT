for i in `seq 223 285`;
do
	cd $i
	ls
	rm -rf VID
	cd ..
done
