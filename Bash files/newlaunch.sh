module load ffmpeg-3.2.2
cd APES2
X=(1111 4444 5423 6654 9111 1122 1337)
: <<'END'
#nohup srun --partition=main --time=2- --mem=4000 python duel.py 599 --exploration 0.1 --tau 0.001 --activation tanh --advantage max --seed 9111 --batch_size 32 --totalsteps 500000 --details "punishment, b:541 duel,nactions: 4" --naction 4 --train_m 541 --target_m 541 --rwrdschem -10 1000 -0.1 >logs/599.out &
#Replication of 541
#counter=538
#X=(1111 4444 5423 6654 9111 1122 1337)

#for t in `seq 0 6`;
#do
#	counter=$((counter+1))
#	echo '#actions:4,seed:'${X[$t]},id:$counter,Duel
#	nohup srun --partition=long,gpu --time=8- --mem=4000 python duel.py $counter --exploration 1.0 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 2000000 --details "Zombie, duel,nactions: 4" --naction 4 --rwrdschem -10 1000 -0.1 >logs/"$counter.out" &
#done

#replication of 599
#counter=1510
#for t in `seq 0 6`;
#do
#	counter=$((counter+1))
#	echo '#actions:4,seed:'${x[$t]},id:$counter,duel
#	nohup srun --partition=long,gpu --time=8- --mem=4000 python duel.py $counter --exploration 0.1 --tau 0.001 --activation tanh --advantage max --seed ${x[$t]} --batch_size 32 --totalsteps 500000 --details "punishment, b:541 duel,nactions: 4" --naction 4 --train_m 541 --target_m 541 --rwrdschem -10 1000 -0.1 >logs/"$counter.out" &
#done

#replication of 599
#counter=1510
#for t in `seq 0 6`;
#do
#	counter=$((counter+1))
#	echo '#actions:4,seed:'${x[$t]},id:$counter,duel
#	nohup srun --partition=long,gpu --time=8- --mem=4000 python duel.py $counter --exploration 0.1 --tau 0.001 --activation tanh --advantage max --seed ${x[$t]} --batch_size 32 --totalsteps 2000000 --details "punishment, b:541 duel,nactions: 4" --naction 4 --train_m 541 --target_m 541 --rwrdschem -10 1000 -0.1 >logs/"$counter.out" &
#done

#nohup srun --partition=main --time=2- --mem=4000 python collect_data.py 1 --totalsteps 2000000 --train_m 601 --naction 4 --samples 200000 >1.out &
#nohup srun --partition=main --time=2- --mem=4000 python collect_data.py 2 --totalsteps 2000000 --train_m 601 --naction 4 --samples 500000 >2.out &
#nohup srun --partition=main --time=2- --mem=8000 python collect_data.py 4 --totalsteps 2000000 --train_m 601 --naction 4 --samples 1000000 >3.out &
counter=903
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#actions:4,seed:'${x[$t]},id:$counter,duel
	nohup srun --partition=main --time=6- --mem=4000 python duel.py $counter --exploration 0.1 --tau 0.001 --activation tanh --advantage max --seed ${x[$t]} --batch_size 32 --totalsteps 1000000 --details "punishment, b:541 duel,nactions: 4" --naction 4 --train_m 541 --target_m 541 --rwrdschem -10 1000 -0.1 >logs/"$counter.out" &
done
END

