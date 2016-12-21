#for i in {78,79,80};
for i in `seq 1 63`;
do
	python generate_TargetModel.py --ID $i
done

