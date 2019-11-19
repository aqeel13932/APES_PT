### Food spawn in all places even at home location.
module load ffmpeg-3.2.2
cd MN_project
X=(1111 4444 5423 6654 9111 1122 1337)
: <<'END'

counter=0
#starting experiment 1-7
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#seed:'${X[$t]},id:$counter,Duel
	nohup srun --partition=main --time=6- --mem=6000  python morning_night.py $counter --exploration 1.0 --tau 0.001 --train_repeat 4 --activation tanh --advantage max --seed ${X[$t]} --batch_size 16 --totalsteps 6000000 --details "End to End,DM + duel," --max_timesteps 80 --svision 360  >logs/"$counter.out" & 
done

# 1000,-100 reward 8-14
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#seed:'${X[$t]},id:$counter,Duel
	nohup srun --partition=main --time=6- --mem=6000  python morning_night.py $counter --exploration 1.0 --tau 0.001 --train_repeat 4 --activation tanh --advantage max --seed ${X[$t]} --batch_size 16 --totalsteps 6000000 --details "End to End,DM + duel," --max_timesteps 80 --svision 360 --rwrdschem 0 1000 -100  >logs/"$counter.out" & 
done

# different reward scheme to balance eating food with night punishment. 15-21
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#seed:'${X[$t]},id:$counter,Duel
	nohup srun --partition=main --time=6- --mem=6000  python morning_night.py $counter --exploration 1.0 --tau 0.001 --train_repeat 4 --activation tanh --advantage max --seed ${X[$t]} --batch_size 16 --totalsteps 6000000 --details "End to End,DM + duel," --max_timesteps 80 --svision 360 --rwrdschem 0 1 -2.5  >logs/"$counter.out" & 
done

### adding a clue 22-28
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#seed:'${X[$t]},id:$counter,Duel
	nohup srun --partition=main --time=6- --mem=6000  python morning_night.py $counter --exploration 1.0 --tau 0.001 --train_repeat 4 --activation tanh --advantage max --seed ${X[$t]} --batch_size 16 --totalsteps 6000000 --details "End to End,DM + duel," --max_timesteps 80 --svision 360 --rwrdschem 0 1 -2.5 --clue  >logs/"$counter.out" & 
done

### No clue, but No food at night 29-35
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#seed:'${X[$t]},id:$counter,Duel
	nohup srun --partition=main --time=6- --mem=6000  python morning_night.py $counter --exploration 1.0 --tau 0.001 --train_repeat 4 --activation tanh --advantage max --seed ${X[$t]} --batch_size 16 --totalsteps 6000000 --details "End to End,DM + duel," --max_timesteps 80 --svision 360 --rwrdschem 0 1 -2.5 --nofood  >logs/"$counter.out" & 
done

### No food at night, with clue 36-42
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#seed:'${X[$t]},id:$counter,Duel
	nohup srun --partition=main --time=6- --mem=6000  python morning_night.py $counter --exploration 1.0 --tau 0.001 --train_repeat 4 --activation tanh --advantage max --seed ${X[$t]} --batch_size 16 --totalsteps 6000000 --details "End to End,DM + duel," --max_timesteps 80 --svision 360 --rwrdschem 0 1 -2.5 --nofood --clue  >logs/"$counter.out" & 
done

### No food at night, reward 1000,-100 43-49
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#seed:'${X[$t]},id:$counter,Duel
	nohup srun --partition=main --time=6- --mem=6000  python morning_night.py $counter --exploration 1.0 --tau 0.001 --train_repeat 4 --activation tanh --advantage max --seed ${X[$t]} --batch_size 16 --totalsteps 6000000 --details "End to End,DM + duel," --max_timesteps 80 --svision 360 --rwrdschem 0 1000 -100 --nofood  >logs/"$counter.out" & 
done

### No Food during night, with clue, Reard: 1000,-100, 50-56
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#seed:'${X[$t]},id:$counter,Duel
	nohup srun --partition=main --time=6- --mem=6000  python morning_night.py $counter --exploration 1.0 --tau 0.001 --train_repeat 4 --activation tanh --advantage max --seed ${X[$t]} --batch_size 16 --totalsteps 6000000 --details "End to End,DM + duel," --max_timesteps 80 --svision 360 --rwrdschem 0 1000 -100 --nofood --clue  >logs/"$counter.out" & 
done

### adding a clue 22-28 " Check of it learn faster or not
counter=56
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#seed:'${X[$t]},id:$counter,Duel
	nohup srun --partition=main --time=6- --mem=6000  python morning_night.py $counter --exploration 1.0 --tau 0.001 --train_repeat 4 --activation tanh --advantage max --seed ${X[$t]} --batch_size 16 --totalsteps 1000000 --details "End to End,DM + duel," --max_timesteps 80 --svision 360 --rwrdschem 0 1 -2.5 --clue --vanish 0.95  >logs/"$counter.out" & 
done
END

counter=63
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#seed:'${X[$t]},id:$counter,Duel
	nohup srun --partition=main --time=6- --mem=6000  python morning_night.py $counter --exploration 1.0 --tau 0.001 --train_repeat 4 --activation tanh --advantage max --seed ${X[$t]} --batch_size 16 --totalsteps 6000000 --details "End to End,DM + duel," --max_timesteps 80 --svision 360 --rwrdschem 0 1 -2.5 --clue --max_timesteps 160 >logs/"$counter.out" & 
done
