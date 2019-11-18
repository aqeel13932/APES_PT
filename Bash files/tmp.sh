

cd APES
counter=13
tmpexp=0.1
tmptau=0.001
for layers in {2,3};
do
	for hidsiz in {64,128};
	do
		echo $counter,$layers,$hidsiz,E24
		#nohup srun --partition=long --time=10- python duel.py $counter --exploration $exp --tau $tau --activation tanh --advantage max --seed 4917 >logs/"$counter.out" &
		let "counter=counter+1"
		sleep 1

		echo $counter,$layers,$hidsiz,E28
		#nohup srun --partition=long --time=10- python duel.py $counter --exploration $exp --tau $tau --activation tanh --advantage max --seed 4917 >logs/"$counter.out" &
		let "counter=counter+1"
		sleep 1

		echo $counter,$layers,$hidsiz,E36
		#nohup srun --partition=long --time=10- python duel.py $counter --exploration $exp --tau $tau --activation tanh --advantage max --seed 4917 >logs/"$counter.out" &
		let "counter=counter+1"
		sleep 1

		#nohup srun --partition=long --time=5- python duel.py $counter --exploration $exp --tau $tau --activation relu --advantage naive --seed 1337 >logs/"$counter.out" &
	done
done
