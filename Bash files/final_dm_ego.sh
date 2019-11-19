module load ffmpeg-3.2.2
cd APES_Ego

#Major update: Using deep mind archeticture (LSTM+ CNN), No last actions provided.
# 8th stage of the new series of experiments where dominant is static, food has a probability mask around dominant, subordinate always on left side looking right. The subordinate get reward per time step when dominant can see food and punishment if dom can't see food. the experiment with 6M steps and subordinate has vision of range 6. 
# do training after every episode
#counter=1092
X=(1111 4444 5423 6654 9111 1122 1337)
counter=1244
# Ego centric L1
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#seed:'${X[$t]},id:$counter,Duel
	nohup srun --partition=long --time=25- --mem=7000  python Fullego_deep_mind_PT_L1.py $counter --exploration 1.0 --tau 0.001 --train_repeat 4 --activation tanh --advantage max --seed ${X[$t]} --batch_size 16 --totalsteps 6000000 --details "End to End,DM + duel," --svision 180 --max_timesteps 100 --rwrdschem 0 1000 -0.1 --Ego >logs/"$counter.out" & 
done
# Ego-centric L2 
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#seed:'${X[$t]},id:$counter,Duel
	nohup srun --partition=long --time=25- --mem=7000  python Fullego_deep_mind_PT_L2_clip.py $counter --exploration 1.0 --tau 0.001 --train_repeat 4 --activation tanh --advantage max --seed ${X[$t]} --batch_size 16 --totalsteps 6000000 --details "End to End,DM + duel," --svision 180 --max_timesteps 100 --rwrdschem 0 1000 -0.1 --Ego >logs/"$counter.out" & 
done
# Ego-centric L3 
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#seed:'${X[$t]},id:$counter,Duel
	nohup srun --partition=long --time=25- --mem=7000  python Fullego_deep_mind_PT_L3.py $counter --exploration 1.0 --tau 0.001 --train_repeat 4 --activation tanh --advantage max --seed ${X[$t]} --batch_size 16 --totalsteps 20000000 --details "End to End,DM + duel," --svision 180 --max_timesteps 100 --rwrdschem 0 1000 -0.1 --Ego >logs/"$counter.out" & 
done
: <<'END'
END
