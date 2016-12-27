#for i in {78,79,80};
for i in `seq 1 63`;
do
	cd $i
	#7za a models.7z MOD model.h5
	rm models.7z
	rm -rf MOD
	#rm -rf MOD
	#rm model.h5
	cd ..
done
	#{18,22,25,26,33,37,38,39,40,50,54,58,59,60};
#do
#	cd E$i/output
#        j=$(ls -d */)
#	mv $j $i
#	mv features.results.out $i.txt
#	7za a ../../output.7z *
#	cd ../..
#done

