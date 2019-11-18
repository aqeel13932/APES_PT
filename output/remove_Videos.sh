for i in `seq 792 903`;
do
	pwd
	cd $i
	ls
	rm -rf VID
	rm -rf PNG
	cd ..
done
