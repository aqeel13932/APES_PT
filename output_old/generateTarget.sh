#for i in {78,79,80};
for i in `seq 1 63`;
do
	cd $i
	rm model.h5
	cd ..
done

