module load ffmpeg-3.2.2
cd APES
# New series of experiments where dominant is static, food static, subordinate always on left side looking right. In all cases sub clear
counter=987
X=(1111 4444 5423 6654 9111 1122 1337)

for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#actions:4,seed:'${X[$t]},id:$counter,Duel
	nohup srun --partition=main --time=6- --mem=4000 python duel_Experiments_V2.py $counter --exploration 1.0 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 2000000 --details "End to End, duel,nactions: 4" --naction 4 --max_timesteps 100 --rwrdschem 0 1000 -0.1 >logs/"$counter.out" &
done
