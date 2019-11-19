module load ffmpeg-3.2.2
cd APES2
X=(1111 4444 5423 6654 9111 1122 1337)

:<<'END'
#Without supervised layer
counter=910
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#actions:4,seed:'${X[$t]},id:$counter,duel
	nohup srun --partition=main --time=6- --mem=4000 python duel_dsfp.py $counter --exploration 0.1 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 1000000 --details "punishment when dom see food, b:541 duel,nactions: 4" --naction 4 --train_m 541 --target_m 541 --rwrdschem -10 1000 -0.1 >logs/"$counter.out" &
done
#counter=917
# With supervised layer trained to classify 2 categories
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#actions:4,seed:'${X[$t]},id:$counter,duel
	nohup srun --partition=main --time=6- --mem=4000 python duel_dsfp_SL.py $counter --exploration 0.1 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 1000000 --details "punishment when dom see food with supervised layer 2 cat, b:541 duel,nactions: 4" --naction 4 --train_m 541 --supervised_m X1_mod_single --rwrdschem -10 1000 -0.1 >logs/"$counter.out" &
done

# With supervised layer trained to classify 3 categories
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#actions:4,seed:'${X[$t]},id:$counter,duel
	nohup srun --partition=main --time=6- --mem=4000 python duel_dsfp_SL.py $counter --exploration 0.1 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 1000000 --details "punishment when dom see food with supervised layer 3 cat, b:541 duel,nactions: 4" --naction 4 --train_m 541 --supervised_m X1_mod_single_3cat --rwrdschem -10 1000 -0.1 >logs/"$counter.out" &
done

# With supervised layer trained to classify 5 categories
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#actions:4,seed:'${X[$t]},id:$counter,duel
	nohup srun --partition=main --time=6- --mem=4000 python duel_dsfp_SL.py $counter --exploration 0.1 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 1000000 --details "punishment when dom see food with supervised layer 5 cat, b:541 duel,nactions: 4" --naction 4 --train_m 541 --supervised_m X1_mod_single_5cat --rwrdschem -10 1000 -0.1 >logs/"$counter.out" &
done

# The punishment is back to normal 
counter=938
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#actions:4,seed:'${X[$t]},id:$counter,duel
	nohup srun --partition=main --time=6- --mem=4000 python duel_dsfp.py $counter --exploration 0.1 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 1000000 --details "b:541 duel,nactions: 4" --naction 4 --train_m 541 --target_m 541 --rwrdschem -10 1000 -0.1 >logs/"$counter.out" &
done
# With supervised layer trained to classify 2 categories
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#actions:4,seed:'${X[$t]},id:$counter,duel
	nohup srun --partition=main --time=6- --mem=4000 python duel_dsfp_SL.py $counter --exploration 0.1 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 1000000 --details "with supervised layer 2 cat, b:541 duel,nactions: 4" --naction 4 --train_m 541 --supervised_m X1_mod_single --rwrdschem -10 1000 -0.1 >logs/"$counter.out" &
done

# With supervised layer trained to classify 3 categories
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#actions:4,seed:'${X[$t]},id:$counter,duel
	nohup srun --partition=main --time=6- --mem=4000 python duel_dsfp_SL.py $counter --exploration 0.1 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 1000000 --details "with supervised layer 3 cat, b:541 duel,nactions: 4" --naction 4 --train_m 541 --supervised_m X1_mod_single_3cat --rwrdschem -10 1000 -0.1 >logs/"$counter.out" &
done

# With supervised layer trained to classify 5 categories
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#actions:4,seed:'${X[$t]},id:$counter,duel
	nohup srun --partition=main --time=6- --mem=4000 python duel_dsfp_SL.py $counter --exploration 0.1 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 1000000 --details "With supervised layer 5 cat, b:541 duel,nactions: 4" --naction 4 --train_m 541 --supervised_m X1_mod_single_5cat --rwrdschem -10 1000 -0.1 >logs/"$counter.out" &
done
iiiii

END

# The punishment is back to normal, and there two layers after SL one 64 and another 16
counter=966
# With supervised layer trained to classify 2 categories
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#actions:4,seed:'${X[$t]},id:$counter,duel
	nohup srun --partition=main --time=6- --mem=4000 python duel_RL_SL_extralayers.py $counter --exploration 0.1 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 1000000 --details "with supervised layer 2 cat, b:541 duel,nactions: 4" --naction 4 --train_m 541 --supervised_m X1_mod_single --rwrdschem -10 1000 -0.1 >logs/"$counter.out" &
done

# With supervised layer trained to classify 3 categories
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#actions:4,seed:'${X[$t]},id:$counter,duel
	nohup srun --partition=main --time=6- --mem=4000 python duel_RL_SL_extralayers.py $counter --exploration 0.1 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 1000000 --details "with supervised layer 3 cat, b:541 duel,nactions: 4" --naction 4 --train_m 541 --supervised_m X1_mod_single_3cat --rwrdschem -10 1000 -0.1 >logs/"$counter.out" &
done

# With supervised layer trained to classify 5 categories
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#actions:4,seed:'${X[$t]},id:$counter,duel
	nohup srun --partition=main --time=6- --mem=4000 python duel_RL_SL_extralayers.py $counter --exploration 0.1 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 1000000 --details "With supervised layer 5 cat, b:541 duel,nactions: 4" --naction 4 --train_m 541 --supervised_m X1_mod_single_5cat --rwrdschem -10 1000 -0.1 >logs/"$counter.out" &
done
