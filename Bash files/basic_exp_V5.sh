module load ffmpeg-3.2.2
cd APES
X=(1111 4444 5423 6654 9111 1122 1337)
: <<'END'
# 5th stage of the new series of experiments where dominant is static, food has a probability mask around dominant, subordinate always on left side looking right. The subordinate get reward per time step when dominant can see food and punishment if dom can't see food. recording is active during test, this experiment with 6M steps and subordinate has vision of range 6

counter=1022
X=(1111 4444 5423 6654 9111 1122 1337)

for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#actions:4,seed:'${X[$t]},id:$counter,Duel
	nohup srun --partition=main --time=6- --mem=4000 python duel_Experiments_V5.py $counter --exploration 1.0 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 6000000 --details "End to End,360 vision, 6 blocks range, duel,nactions: 4" --naction 4 --svision 360 --max_timesteps 100 --rwrdschem 0 1000 -0.1 >logs/"$counter.out" &
done

# Trying different taus and optimizers.

counter=1036
ta=(0.01 0.1)
for tau in "${ta[@]}";
do
	for t in `seq 0 6`;
	do
		counter=$((counter+1))
		echo '#actions:4,seed:'${X[$t]},id:$counter,Duel,$tau
		nohup srun --partition=main --time=6- --mem=4000 python duel_Experiments_V5.py $counter --exploration 1.0 --tau $tau --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 6000000 --details "End to End,360 vision, 6 blocks range, duel,nactions: 4" --naction 4 --svision 360 --max_timesteps 100 --rwrdschem 0 1000 -0.1 >logs/"$counter.out" &
	done
done


ta=(0.001 0.01 0.1)
for tau in "${ta[@]}";
do
	for t in `seq 0 6`;
	do
		counter=$((counter+1))
		echo '#actions:4,seed:'${X[$t]},id:$counter,Duel,$tau
		nohup srun --partition=main --time=6- --mem=4000 python duel_Experiments_V5.py $counter --exploration 1.0 --tau $tau --optimizer 'rmsprop' --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 6000000 --details "End to End,360 vision, 6 blocks range, duel,nactions: 4" --naction 4 --svision 360 --max_timesteps 100 --rwrdschem 0 1000 -0.1 >logs/"$counter.out" &
	done
done

counter=1071
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#actions:4,seed:'${X[$t]},id:$counter,Duel
	nohup srun --partition=main --time=6- --mem=4000 python duel_Experiments_V5.py $counter --exploration 1.0 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 20000000 --details "End to End,360 vision, 6 blocks range, duel,nactions: 4" --naction 4 --svision 360 --max_timesteps 100 --rwrdschem 0 1000 -0.1 >logs/"$counter.out" &
done
END
# train for additional 20M 
counter=1078
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#actions:4,seed:'${X[$t]},id:$counter,Duel
	nohup srun --partition=main --time=6- --mem=4000 python duel_Experiments_V5.py $counter --exploration 0.1 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 20000000 --details "End to End,360 vision, 6 blocks range, duel,nactions: 4" --naction 4 --svision 360 --max_timesteps 100 --train_m 1073 --rwrdschem 0 1000 -0.1 >logs/"$counter.out" &
done
