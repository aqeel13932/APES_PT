#Stage 1 Experiments to select: activation and advantge.
#cd APES
#counter=13
#for adv in {'naive','avg','max'};
#do
#	for act in {'relu','tanh'};
#	do
#		for s in {1337,4917};
#		do
#			echo $counter,$adv,$act,$s
#			nohup srun --partition=long --time=5- python duel.py $counter --activation relu --advantage $adv --seed $s >logs/"$counter.out" &
#			let "counter=counter+1"
#			sleep 1
#		done
#	done
#done

#########Stage 2.1###### Choose Exploration and tau
#cd APES
#counter=13
#tmpexp=0.1
#tmptau=0.001

#for loop for exploration
#for loop for tau
#condition for exploration and tau
#do P1 to P4

#########Stage 2.2###### Choose number of layers and hidden size
#cd APES
#counter=57
#for layers in {2,3};
#do
#	for hidsiz in {64,128};
#	do
#		echo $counter,$layers,$hidsiz,E24
#		nohup srun --partition=long --time=10- python duel.py $counter --exploration 0.1 --tau 0.0001 --activation tanh --advantage max --seed 4917 --hidden_size $hidsiz --layers $layers >logs/"$counter.out" &
#		let "counter=counter+1"
#		sleep 1

#		echo $counter,$layers,$hidsiz,E28
#		nohup srun --partition=long --time=10- python duel.py $counter --exploration 0.01 --tau 0.1 --activation tanh --advantage max --seed 4917 --hidden_size $hidsiz --layers $layers >logs/"$counter.out" &
#		let "counter=counter+1"
#		sleep 1

#		echo $counter,$layers,$hidsiz,E36
#		nohup srun --partition=long --time=10- python duel.py $counter --exploration 0.01 --tau 0.001 --activation tanh --advantage max --seed 4917 --hidden_size $hidsiz --layers $layers >logs/"$counter.out" &
#		let "counter=counter+1"
#		sleep 1
#	done
#done

######Stage 2.3 #### more about layers and hidden size

#cd APES
#counter=69
#for layers in {1,};
#do
#	for hidsiz in {64,128};
#	do
#		echo $counter,$layers,$hidsiz,E24
#		nohup srun --partition=long --time=10- python duel.py $counter --exploration 0.1 --tau 0.0001 --activation tanh --advantage max --seed 4917 --hidden_size $hidsiz --layers $layers >logs/"$counter.out" &
#		let "counter=counter+1"
#		sleep 1
#
#		echo $counter,$layers,$hidsiz,E28
#		nohup srun --partition=long --time=10- python duel.py $counter --exploration 0.01 --tau 0.1 --activation tanh --advantage max --seed 4917 --hidden_size $hidsiz --layers $layers >logs/"$counter.out" &
#		let "counter=counter+1"
#		sleep 1
#
#		echo $counter,$layers,$hidsiz,E36
#		nohup srun --partition=long --time=10- python duel.py $counter --exploration 0.01 --tau 0.001 --activation tanh --advantage max --seed 4917 --hidden_size $hidsiz --layers $layers >logs/"$counter.out" &
#		let "counter=counter+1"
#		sleep 1
#	done
#done
#
#
#for layers in {2,3};
#do
#	for hidsiz in {100,};
#	do
#		echo $counter,$layers,$hidsiz,E24
#		nohup srun --partition=long --time=10- python duel.py $counter --exploration 0.1 --tau 0.0001 --activation tanh --advantage max --seed 4917 --hidden_size $hidsiz --layers $layers >logs/"$counter.out" &
#		let "counter=counter+1"
#		sleep 1
#
#		echo $counter,$layers,$hidsiz,E28
#		nohup srun --partition=long --time=10- python duel.py $counter --exploration 0.01 --tau 0.1 --activation tanh --advantage max --seed 4917 --hidden_size $hidsiz --layers $layers >logs/"$counter.out" &
#		let "counter=counter+1"
#		sleep 1
#
#		echo $counter,$layers,$hidsiz,E36
#		nohup srun --partition=long --time=10- python duel.py $counter --exploration 0.01 --tau 0.001 --activation tanh --advantage max --seed 4917 --hidden_size $hidsiz --layers $layers >logs/"$counter.out" &
#		let "counter=counter+1"
#		sleep 1
#	done
#done

###### Stage 3#### batch size and reply size
#cd APES
#counter=81
#
#for basi in {10,16,32,64,128,256};
#do
#	for resi in {100000,200000,300000,400000,500000};
#	do
#		
#		if [ "$basi" == 10 ] && [ "$resi" == 100000 ] ;
#		then
#			echo bad 
#		else
#			echo $counter,$basi,$resi,E24
#			nohup srun --partition=long --mem=10000 --time=10- python duel.py $counter --exploration 0.1 --tau 0.0001 --activation tanh --advantage max --seed 4917 --batch_size $basi --replay_size $resi >logs/"$counter.out" &
#			let "counter=counter+1"
#			sleep 1
#
#			echo $counter,$basi,$resi,E28
#			nohup srun --partition=long --mem=10000 --time=10- python duel.py $counter --exploration 0.01 --tau 0.1 --activation tanh --advantage max --seed 4917 --batch_size $basi --replay_size $resi >logs/"$counter.out" &
#			let "counter=counter+1"
#			sleep 1
#
#			echo $counter,$basi,$resi,E36
#			nohup srun --partition=long --mem=10000 --time=10- python duel.py $counter --exploration 0.01 --tau 0.001 --activation tanh --advantage max --seed 4917 --batch_size $basi --replay_size $resi >logs/"$counter.out" &
#			let "counter=counter+1"
#			sleep 1
#		fi
#	done
#done

###### Stage 3.1#### train repeate
cd APES
counter=168

for tr in {2,4,8,16};
do
		
	echo $counter,$tr,E24
	nohup srun --partition=long --time=10- python duel.py $counter --exploration 0.1 --tau 0.0001 --activation tanh --advantage max --seed 4917 --train_repeat $tr >logs/"$counter.out" &
	let "counter=counter+1"
	sleep 1

	echo $counter,$tr,E36
	nohup srun --partition=long --time=10- python duel.py $counter --exploration 0.01 --tau 0.001 --activation tanh --advantage max --seed 4917 --train_repeat $tr >logs/"$counter.out" &
	let "counter=counter+1"
	sleep 1

	echo $counter,$tr,E108
	nohup srun --partition=long --time=10- python duel.py $counter --exploration 0.1 --tau 0.0001 --activation tanh --advantage max --seed 4917 --batch_size 32 --train_repeat $tr >logs/"$counter.out" &
	let "counter=counter+1"
	sleep 1

	echo $counter,$tr,E110
	nohup srun --partition=long --time=10- python duel.py $counter --exploration 0.01 --tau 0.001 --activation tanh --advantage max --seed 4917 --batch_size 32 --train_repeat $tr >logs/"$counter.out" &
	let "counter=counter+1"
	sleep 1
done
