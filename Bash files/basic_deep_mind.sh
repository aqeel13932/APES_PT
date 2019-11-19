module load ffmpeg-3.2.2
cd APES

#Major update: Using deep mind archeticture (LSTM+ CNN), No last actions provided.
# 8th stage of the new series of experiments where dominant is static, food has a probability mask around dominant, subordinate always on left side looking right. The subordinate get reward per time step when dominant can see food and punishment if dom can't see food. the experiment with 6M steps and subordinate has vision of range 6. 
# do training after every episode
#counter=1092
X=(1111 4444 5423 6654 9111 1122 1337)
: <<'END'
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#actions:4,seed:'${X[$t]},id:$counter,Duel
	nohup srun --partition=main --time=6- --mem=6000 python deep_mind_PT.py $counter --exploration 1.0 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 16 --totalsteps 6000000 --details "End to End,360 vision, 6 blocks range, duel,nactions: 4" --svision 360 --max_timesteps 100 --rwrdschem 0 1000 -0.1 >logs/"$counter.out" &
done

#same as before, but do training once every time step

counter=1099
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#actions:4,seed:'${X[$t]},id:$counter,Duel
	nohup srun --partition=main --time=6- --mem=6000 python deep_mind_PT.py $counter --exploration 1.0 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 16 --totalsteps 6000000 --details "End to End,360 vision, 6 blocks range, duel,nactions: 4" --svision 360 --max_timesteps 100 --rwrdschem 0 1000 -0.1 >logs/"$counter.out" &
done


# Same as before but we train 8 times after every episode. 
counter=1106
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#actions:4,seed:'${X[$t]},id:$counter,Duel
	nohup srun --partition=main --time=6- --mem=6000 python deep_mind_PT.py $counter --exploration 1.0 --tau 0.001 --train_repeat 8 --activation tanh --advantage max --seed ${X[$t]} --batch_size 16 --totalsteps 6000000 --details "End to End,360 vision, 6 blocks range, duel,nactions: 4" --svision 360 --max_timesteps 100 --rwrdschem 0 1000 -0.1 >logs/"$counter.out" &
done

# Same as before but without reward shapping. 
counter=1113
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#actions:4,seed:'${X[$t]},id:$counter,Duel
	nohup srun --partition=main --time=6- --mem=6000 python deep_mind_PT_new_reward.py $counter --exploration 1.0 --tau 0.001 --train_repeat 4 --activation tanh --advantage max --seed ${X[$t]} --batch_size 16 --totalsteps 6000000 --details "End to End,360 vision, 6 blocks range, duel,nactions: 4" --svision 360 --max_timesteps 100 --rwrdschem 0 1000 -0.1 >logs/"$counter.out" &   
done
# same as before but with 20M steps
counter=1120
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#actions:4,seed:'${X[$t]},id:$counter,Duel
	nohup srun --partition=main --time=6- --mem=6000 python deep_mind_PT_new_reward.py $counter --exploration 1.0 --tau 0.001 --train_repeat 4 --activation tanh --advantage max --seed ${X[$t]} --batch_size 16 --totalsteps 20000000 --details "End to End,360 vision, 6 blocks range, duel,nactions: 4" --svision 360 --max_timesteps 100 --rwrdschem 0 1000 -0.1 >logs/"$counter.out" &
done

counter=1127
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#actions:4,seed:'${X[$t]},id:$counter,Duel
	nohup srun --partition=main --time=6- --mem=6000 python deep_mind_PT_new_reward.py $counter --exploration 1.0 --tau 0.001 --train_repeat 4 --activation tanh --advantage max --seed ${X[$t]} --batch_size 16 --totalsteps 6000000 --details "End to End,360 vision, 6 blocks range, duel,nactions: 4" --svision 180 --max_timesteps 100 --rwrdschem 0 1000 -0.1 >logs/"$counter.out" &   
done
# deep mind archeticture with 6X6 vision range
counter=1134
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#actions:4,seed:'${X[$t]},id:$counter,Duel
	nohup srun --partition=main --time=6- --mem=6000 python deep_mind_PT_new_reward.py $counter --exploration 1.0 --tau 0.001 --train_repeat 4 --activation tanh --advantage max --seed ${X[$t]} --batch_size 16 --totalsteps 6000000 --details "End to End,6 blocks range,DM+duel" --svision 360 --max_timesteps 100 --rwrdschem 0 1000 -0.1 >logs/"$counter.out" &
done

# deep mind archeticture with dominant random position + Full range but without obestacles.
counter=1142
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#seed:'${X[$t]},id:$counter,Duel
	nohup srun --partition=main --time=6- --mem=6000  python deep_mind_PT_L3.py $counter --exploration 1.0 --tau 0.001 --train_repeat 4 --activation tanh --advantage max --seed ${X[$t]} --batch_size 16 --totalsteps 6000000 --details "End to End,DM + duel," --svision 180 --max_timesteps 100 --rwrdschem 0 1000 -0.1 >logs/"$counter.out" & 
done

# try different taus for complexity level 2
counter=1151
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#seed:'${X[$t]},id:$counter,Duel
	nohup srun --partition=main --time=6- --mem=6000  python deep_mind_PT_new_reward.py $counter --exploration 1.0 --tau 0.0005 --train_repeat 4 --activation tanh --advantage max --seed ${X[$t]} --batch_size 16 --totalsteps 6000000 --details "End to End,DM + duel," --svision 180 --max_timesteps 100 --rwrdschem 0 1000 -0.1 >logs/"$counter.out" & 
done
# try different taus for coplexity level 2
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#seed:'${X[$t]},id:$counter,Duel
	nohup srun --partition=main --time=6- --mem=6000  python deep_mind_PT_new_reward.py $counter --exploration 1.0 --tau 0.01 --train_repeat 4 --activation tanh --advantage max --seed ${X[$t]} --batch_size 16 --totalsteps 6000000 --details "End to End,DM + duel," --svision 180 --max_timesteps 100 --rwrdschem 0 1000 -0.1 >logs/"$counter.out" & 
done
# try clipping gradient for complexity level 2
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#seed:'${X[$t]},id:$counter,Duel
	nohup srun --partition=main --time=6- --mem=6000  python deep_mind_PT_L2_clip.py $counter --exploration 1.0 --tau 0.001 --train_repeat 4 --activation tanh --advantage max --seed ${X[$t]} --batch_size 16 --totalsteps 6000000 --details "End to End,DM + duel," --svision 180 --max_timesteps 100 --rwrdschem 0 1000 -0.1 >logs/"$counter.out" & 
done
# Level 3 with clipping
counter=1172
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#seed:'${X[$t]},id:$counter,Duel
	nohup srun --partition=main --time=6- --mem=6000  python deep_mind_PT_L3.py $counter --exploration 1.0 --tau 0.001 --train_repeat 4 --activation tanh --advantage max --seed ${X[$t]} --batch_size 16 --totalsteps 6000000 --details "End to End,DM + duel," --svision 180 --max_timesteps 100 --rwrdschem 0 1000 -0.1 >logs/"$counter.out" & 
done

counter=1179
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#seed:'${X[$t]},id:$counter,Duel
	nohup srun --partition=long --time=20- --mem=6000  python deep_mind_PT_L3.py $counter --exploration 1.0 --tau 0.001 --train_repeat 4 --activation tanh --advantage max --seed ${X[$t]} --batch_size 16 --totalsteps 20000000 --details "End to End,DM + duel," --svision 180 --max_timesteps 100 --rwrdschem 0 1000 -0.1 >logs/"$counter.out" & 
done

END
# Level 3 complixy with orientations feeded to CNN 
counter=1216
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#seed:'${X[$t]},id:$counter,Duel
	nohup srun --partition=long --time=20- --mem=6000  python AllCNN_deep_mind_PT_L3.py $counter --exploration 1.0 --tau 0.001 --train_repeat 4 --activation tanh --advantage max --seed ${X[$t]} --batch_size 16 --totalsteps 20000000 --details "End to End,DM + duel," --svision 180 --max_timesteps 100 --rwrdschem 0 1000 -0.1 >logs/"$counter.out" & 
done
