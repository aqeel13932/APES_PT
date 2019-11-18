for counter in `seq 11 20`;
do
	echo $counter
	nohup srun --partition=long --time=10- --mem=5000 python Dataset_Supervised.py $counter >"$counter.out" &
done
