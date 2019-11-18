#### Phase 3, AI agent with zombie agent (nothing but the input of the other agent)#####
# trying the settings with the best 
:<<'Details'
Phase 3, AI agent with zombi agent (nothing but extra agent input for the network)
trying to check the hyperparameters with the new agent input.
Details
:<<'ACCOMPLISHED'

cd APES
counter=185
nohup srun --partition=long --time=10- --mem=5000 python duel.py $counter --exploration 0.01 --tau 0.001 --activation tanh --advantage max --seed 4917 --batch_size 32 --totalsteps 1100000 --details "zombie agent" >logs/"$counter.out" &

counter=186
nohup srun --partition=long --time=10- --mem=5000 python duel.py $counter --exploration 0.01 --tau 0.001 --activation tanh --advantage max --seed 4917 --batch_size 32 --totalsteps 2000000 --details "zombie agent" >logs/"$counter.out" &
#cd APES
#counter=187
#nohup run --partition=long --time=10- --mem=8000 python duel.py $counter --exploration 0.01 --tau 0.001 --activation tanh --advantage max --seed 4917 --batch_size 32 --totalsteps 500000 --train_m 186 --target_m 186 --details "A*,180,FR,E186" >logs/"$counter.out" &

#counter=188
#nohup srun --partition=long --time=10- --mem=8000 python duel.py $counter --exploration 0.01 --tau 0.001 --activation tanh --advantage max --seed 4917 --batch_size 32 --totalsteps 1100000 --details "A*,180,FR,SCRCH" >logs/"$counter.out" &


cd APES
counter=187
nohup srun --partition=long --time=10- --mem=5000 python duel.py $counter --exploration 0.01 --tau 0.001 --activation tanh --advantage max --seed 13577 --batch_size 32 --totalsteps 1100000 --details "zombie agent" >logs/"$counter.out" &
counter=188
nohup srun --partition=long --time=10- --mem=5000 python duel.py $counter --exploration 0.01 --tau 0.001 --activation tanh --advantage max --seed 1111  --batch_size 32 --totalsteps 1100000 --details "zombie agent" >logs/"$counter.out" &
counter=189
nohup srun --partition=long --time=10- --mem=5000 python duel.py $counter --exploration 0.01 --tau 0.001 --activation tanh --advantage max --seed 4444 --batch_size 32 --totalsteps 1100000 --details "zombie agent" >logs/"$counter.out" &
counter=190
nohup srun --partition=long --time=10- --mem=5000 python duel.py $counter --exploration 0.01 --tau 0.001 --activation tanh --advantage max --seed 5423 --batch_size 32 --totalsteps 1100000 --details "zombie agent" >logs/"$counter.out" &
counter=191
nohup srun --partition=long --time=10- --mem=5000 python duel.py $counter --exploration 0.01 --tau 0.001 --activation tanh --advantage max --seed 6654 --batch_size 32 --totalsteps 1100000 --details "zombie agent" >logs/"$counter.out" &
counter=192
nohup srun --partition=long --time=10- --mem=5000 python duel.py $counter --exploration 0.01 --tau 0.001 --activation tanh --advantage max --seed 9111 --batch_size 32 --totalsteps 1100000 --details "zombie agent" >logs/"$counter.out" &
counter=193
nohup srun --partition=long --time=10- --mem=5000 python duel.py $counter --exploration 0.01 --tau 0.001 --activation tanh --advantage max --seed 1122 --batch_size 32 --totalsteps 1100000 --details "zombie agent" >logs/"$counter.out" &


cd APES
X=(1111 4444 5423 6654 9111 1122 1337)
for t in `seq 0 6`;
do
	echo ${X[$t]}
	counter=$((t +194))
	echo $counter

	nohup srun --partition=long --time=10- --mem=5000 python duel.py $counter --exploration 0.02 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 1100000 --details "zombie agent" >logs/"$counter.out" &
done

cd APES
X=(1111 4444 5423 6654 9111 1122 1337)
for t in `seq 0 6`;
do
	echo ${X[$t]}
	counter=$((t +201))
	echo $counter

	nohup srun --partition=long --time=10- --mem=5000 python duel.py $counter --exploration 0.01 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 2000000 --details "zombie agent" >logs/"$counter.out" &
done


cd APES
X=(1111 4444 5423 6654 9111 1122 1337 4917)
for t in `seq 0 7`;
do
	echo ${X[$t]}
	counter=$((t +208))
	echo $counter

	nohup srun --partition=long --time=10- --mem=5000 python duel.py $counter --exploration 0.01 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 500000 --details "zombie agent" --rwrdschem -100 1000 -0.1 --train_m 186 --target_m 186 >logs/"$counter.out" &
done


cd APES2
module load ffmpeg-3.2.2
counter=217
echo $counter
nohup srun --partition=long --time=10- --mem=15000 python duel.py $counter --exploration 0.01 --tau 0.001 --activation tanh --advantage max --seed 4917 --batch_size 32 --totalsteps 30000000 --details "Crazy Experiment" --rwrdschem -100 1000 -0.1 --layers 10 >logs/"$counter.out" &

cd APES2
module load ffmpeg-3.2.2
counter=218
echo $counter
nohup srun --partition=long --time=10- --mem=15000 python duel.py $counter --exploration 1 --tau 0.001 --activation tanh --advantage max --seed 4917 --batch_size 32 --totalsteps 1000000 --details "Crazy Experiment" --rwrdschem -100 1000 -0.1 --layers 10 >logs/"$counter.out" &

cd APES
module load ffmpeg-3.2.2
X=(1111 4444 5423 6654 9111 1122 1337 4917)
for t in `seq 0 7`;
do
	echo ${X[$t]}
	counter=$((t +227))
	echo $counter

	nohup srun --partition=long --time=10- --mem=5000 python duel.py $counter --exploration 0.01 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 500000 --details "punishment on meet based on model 187" --rwrdschem -10 1000 -0.1 --train_m 187 --target_m 187 >logs/"$counter.out" &
done

# Calculate with less nodes and more layers
cd APES2
module load ffmpeg-3.2.2
X=(1111 4444 5423 6654 9111 1122 1337 4917)
for t in `seq 0 7`;
do
	echo ${X[$t]}
	counter=$((t +251))
	echo $counter

	nohup srun --partition=long --time=10- --mem=5000 python duel.py $counter --exploration 1 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 2000000 --details "more layers less hidden nodes." --rwrdschem -10 1000 -0.1 --hidden_size 32 --layers 3 >logs/"$counter.out" &
done
# Calculate with less nodes and more layers
cd APES2
module load ffmpeg-3.2.2
X=(1111 4444 5423 6654 9111 1122 1337 4917)
for t in `seq 0 7`;
do
	echo ${X[$t]}
	counter=$((t +259))
	echo $counter

	nohup srun --partition=long --time=10- --mem=5000 python duel.py $counter --exploration 1 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 4000000 --details "more layers less hidden nodes." --rwrdschem -10 1000 -0.1 --hidden_size 32 --layers 3 >logs/"$counter.out" &
done
# With punishment based on 207
cd APES
module load ffmpeg-3.2.2
X=(1111 4444 5423 6654 9111 1122 1337 4917)
for t in `seq 0 7`;
do
	echo ${X[$t]}
	counter=$((t +267))
	echo $counter

	nohup srun --partition=long --time=10- --mem=5000 python duel.py $counter --exploration 1 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 500000 --details "two Players based on 207 with punishment." --rwrdschem -10 1000 -0.1 --train_m 207 --target_m 207 >logs/"$counter.out" &
done

# Competitive based on 207
#cd APES
module load ffmpeg-3.2.2
X=(1111 4444 5423 6654 9111 1122 1337 4917)
for t in `seq 0 7`;
do
	echo ${X[$t]}
	counter=$((t +275))
	echo $counter

	nohup srun --partition=long --time=10- --mem=5000 python duel.py $counter --exploration 1 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 500000 --details "two Players based on 207 with punishment." --rwrdschem 0 1000 -0.1 --train_m 207 --target_m 207 >logs/"$counter.out" &
done
# With punishment based on 207 and Dominante Control Range 1
cd APES
module load ffmpeg-3.2.2
X=(1111 4444 5423 6654 9111 1122 1337 4917)
for t in `seq 0 7`;
do
	echo ${X[$t]}
	counter=$((t +283))
	echo $counter

	nohup srun --partition=long --time=10- --mem=5000 python duel.py $counter --exploration 1 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 500000 --details "based on 207-punishment-Dominante control range 1" --rwrdschem -10 1000 -0.1 --train_m 207 --target_m 207 >logs/"$counter.out" &
done
# Calculate with less nodes and more layers
cd APES2
module load ffmpeg-3.2.2
X=(1111 4444 5423 6654 9111 1122 1337 4917)
for t in `seq 0 7`;
do
	echo ${X[$t]}
	counter=$((t +291))
	echo $counter

	nohup srun --partition=long --time=10- --mem=5000 python duel.py $counter --exploration 1 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 2000000 --details "more layers less hidden nodes." --rwrdschem -10 1000 -0.1 --hidden_size 40 --layers 3 >logs/"$counter.out" &
done

# Finding the food with multiplex 2
cd APES
module load ffmpeg-3.2.2
X=(1111 4444 5423 6654 9111 1122 1337 4917)
for t in `seq 0 7`;
do
	echo ${X[$t]}
	counter=$((t +299))
	echo $counter

	nohup srun --partition=long --time=10- --mem=5000 python duel.py $counter --exploration 1 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 2000000 --details "one agent,multiplex 2" --rwrdschem -10 1000 -0.1 >logs/"$counter.out" &
done

# Finding the food with multiplex 3
cd APES3
module load ffmpeg-3.2.2
X=(1111 4444 5423 6654 9111 1122 1337 4917)
for t in `seq 0 7`;
do
	echo ${X[$t]}
	counter=$((t +307))
	echo $counter

	nohup srun --partition=long --time=10- --mem=5000 python duel.py $counter --exploration 1 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 2000000 --details "one agent,multiplex 2" --rwrdschem -10 1000 -0.1 >logs/"$counter.out" &
done


# Finding the food with multiplex 4
cd APES4
module load ffmpeg-3.2.2
X=(1111 4444 5423 6654 9111 1122 1337 4917)
for t in `seq 0 7`;
do
	echo ${X[$t]}
	counter=$((t +315))
	echo $counter

	nohup srun --partition=long --time=10- --mem=5000 python duel.py $counter --exploration 1 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 2000000 --details "one agent,multiplex 2" --rwrdschem -10 1000 -0.1 >logs/"$counter.out" &
done

# Competitive based on 207
cd APES
module load ffmpeg-3.2.2
X=(1111 4444 5423 6654 9111 1122 1337 4917)
for t in `seq 0 7`;
do
	echo ${X[$t]}
	counter=$((t +323))
	echo $counter

	nohup srun --partition=phi --time=8- --mem=5000 python duel.py $counter --exploration 0.01 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 500000 --details "two Players based on 207 with competitive." --rwrdschem 0 1000 -0.1 --train_m 207 --target_m 207 >logs/"$counter.out" &
done

# With punishment based on 207
X=(1111 4444 5423 6654 9111 1122 1337 4917)
for t in `seq 0 7`;
do
	echo ${X[$t]}
	counter=$((t +331))
	echo $counter

	nohup srun --partition=phi --time=8- --mem=5000 python duel.py $counter --exploration 0.01 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 500000 --details "two Players based on 207 with punishment." --rwrdschem -10 1000 -0.1 --train_m 207 --target_m 207 >logs/"$counter.out" &
done

# Finding the food with multiplex 2 Exploration 0.01
cd APES3
module load ffmpeg-3.2.2
X=(1111 4444 5423 6654 9111 1122 1337 4917)
for t in `seq 0 7`;
do
	echo ${X[$t]}
	counter=$((t +339))
	echo $counter

	nohup srun --partition=phi --time=8- --mem=5000 python duel.py $counter --exploration 0.01 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 2000000 --details "one agent,multiplex 2" --rwrdschem -10 1000 -0.1 >logs/"$counter.out" &
done
counter=347
module load ffmpeg-3.2.2
cd AXEL
nohup srun --partition=phi --time=8- --mem=5000 python duel.py $counter --exploration 0.01 --tau 0.001 --activation tanh --advantage max --seed 4917 --batch_size 32 --totalsteps 2000000 --details "Multiplex Axel copy, 2 Frames." >logs/"$counter.out" &

counter=348
module load ffmpeg-3.2.2
cd APES3
nohup srun --partition=phi --time=8- --mem=5000 python duel.py $counter --exploration 0.01 --tau 0.001 --activation tanh --advantage max --seed 4917 --batch_size 32 --totalsteps 2000000 --details "Multiplex Axel copy, 2 Frames.Probabilities Precision is low,PushFront" >logs/"$counter.out" &

counter=349
module load ffmpeg-3.2.2
cd APES3
nohup srun --partition=phi --time=8- --mem=5000 python duel.py $counter --exploration 0.01 --tau 0.001 --activation tanh --advantage max --seed 4917 --batch_size 32 --totalsteps 2000000 --details "Multiplex Axel copy, 2 Frames." >logs/"$counter.out" &
#Replication of experiments E201->207 (best results available)
cd APES
module load ffmpeg-3.2.2
X=(1111 4444 5423 6654 9111 1122 1337)
for t in `seq 0 6`;
do
	echo ${X[$t]}
	counter=$((t +349))
	echo $counter

	nohup srun --partition=phi --time=5- --mem=5000 python duel.py $counter --exploration 0.01 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 2000000 --details "zombie agent, replicating 201->207" >logs/"$counter.out" &
done

#Competitive based on 355
cd APES
module load ffmpeg-3.2.2
X=(1111 4444 5423 6654 9111 1122 1337)
for t in `seq 0 6`;
do
	echo ${X[$t]}
	counter=$((t +356))
	echo $counter

	nohup srun --partition=phi --cpus-per-task=2 --time=5- --mem=5000 python duel.py $counter --exploration 0.01 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 500000 --train_m 355 --target_m 355 --details "competitive based on E355" --rwrdschem 0 1000 -0.1 >logs/"$counter.out" &
done

#Competitive based on 355 , Exp:1
for t in `seq 0 6`;
do
	echo ${X[$t]}
	counter=$((t +363))
	echo $counter

	nohup srun --partition=phi --cpus-per-task=2 --time=5- --mem=5000 python duel.py $counter --exploration 1.0 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 500000 --train_m 355 --target_m 355 --details "competitive based on E355=E207" --rwrdschem 0 1000 -0.1 >logs/"$counter.out" &
done

#Punishment based on 355
for t in `seq 0 6`;
do
	echo ${X[$t]}
	counter=$((t +370))
	echo $counter

	nohup srun --partition=phi --cpus-per-task=2 --time=5- --mem=5000 python duel.py $counter --exploration 0.01 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 500000 --train_m 355 --target_m 355 --details "punishment based on E355" --rwrdschem -10 1000 -0.1 >logs/"$counter.out" &
done

#Punishment based on 355, Exp:1
for t in `seq 0 6`;
do
	echo ${X[$t]}
	counter=$((t +377))
	echo $counter

	nohup srun --partition=phi --cpus-per-task=2 --time=5- --mem=5000 python duel.py $counter --exploration 1.0 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 500000 --train_m 355 --target_m 355 --details "punishment based on E355" --rwrdschem -10 1000 -0.1 >logs/"$counter.out" &
done

cd APES
module load ffmpeg-3.2.2
X=(1111 4444 5423 6654 9111 1122 1337)
#Punishment based on 355
for t in `seq 0 6`;
do
	echo ${X[$t]}
	counter=$((t +384))
	echo $counter

	nohup srun --partition=phi --cpus-per-task=2 --time=5- --mem=5000 python duel.py $counter --exploration 0.01 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 500000 --train_m 355 --target_m 355 --details "punishment based on E355, CR:1" --rwrdschem -10 1000 -0.1 >logs/"$counter.out" &
done

#Punishment based on 355, Exp:1
for t in `seq 0 6`;
do
	echo ${X[$t]}
	counter=$((t +391))
	echo $counter

	nohup srun --partition=phi --cpus-per-task=2 --time=5- --mem=5000 python duel.py $counter --exploration 1.0 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 500000 --train_m 355 --target_m 355 --details "punishment based on E355 CR:1" --rwrdschem -10 1000 -0.1 >logs/"$counter.out" &
done

cd APES
module load ffmpeg-3.2.2
X=(1111 4444 5423 6654 9111 1122 1337)
#Punishment based on 360
for t in `seq 0 6`;
do
	#echo ${X[$t]}
	counter=$((t +398))
	echo $counter

	nohup srun --partition=phi --cpus-per-task=2 --time=5- --mem=5000 python duel.py $counter --exploration 0.01 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 500000 --train_m 360 --target_m 360 --details "punishment based on E360 3rd tier CR:1" --rwrdschem -10 1000 -0.1 >logs/"$counter.out" &
done

#Punishment based on 360, Exp:1
for t in `seq 0 6`;
do
	#echo ${X[$t]}
	counter=$((t +405))
	echo $counter

	nohup srun --partition=phi --cpus-per-task=2 --time=5- --mem=5000 python duel.py $counter --exploration 1.0 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 500000 --train_m 360 --target_m 360 --details "punishment based on E360 3rd tier CR:1" --rwrdschem -10 1000 -0.1 >logs/"$counter.out" &
done

#Punishment based on model 350
X=(1111 4444 5423 6654 9111 1122 1337)
#Punishment based on 355
for t in `seq 0 6`;
do
	#echo ${X[$t]}
	counter=$((t +412))
	echo $counter

	nohup srun --partition=phi --cpus-per-task=2 --time=5- --mem=5000 python duel.py $counter --exploration 0.01 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 500000 --train_m 350 --target_m 350 --details "punishment based on E350, CR:1" --rwrdschem -10 1000 -0.1 >logs/"$counter.out" &
done

#Punishment based on 350, Exp:1
for t in `seq 0 6`;
do
	#echo ${X[$t]}
	counter=$((t +419))
	echo $counter

	nohup srun --partition=phi --cpus-per-task=2 --time=5- --mem=5000 python duel.py $counter --exploration 1.0 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 500000 --train_m 350 --target_m 350 --details "punishment based on E350 CR:1" --rwrdschem -10 1000 -0.1 >logs/"$counter.out" &
done



#Replication of experiments E201->207 (best results available)
cd APES
module load ffmpeg-3.2.2
X=(1111 4444 5423 6654 9111 1122 1337)
for t in `seq 0 6`;
do
	echo ${X[$t]}
	counter=$((t +426))
	echo $counter

	nohup srun --partition=phi --time=5- --mem=5000 python duel_v2.py $counter --exploration 1.0 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 2000000 --details "zombie , newway exp1-0.1 tst 0.05" >logs/"$counter.out" &
done

#Replication of experiments E201->207 (best results available) this time much more time steps.
cd APES
module load ffmpeg-3.2.2
X=(1111 4444 5423 6654 9111 1122 1337)
for t in `seq 0 6`;
do
	echo ${X[$t]}
	counter=$((t +433))
	echo $counter

	nohup srun --partition=phi --time=5- --mem=5000 python duel_v2.py $counter --exploration 1.0 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 6000000 --details "zombie , newway exp1-0.1 tst 0.05" >logs/"$counter.out" &
done

cd APES
module load ffmpeg-3.2.2
X=(1111 4444 5423 6654 9111 1122 1337)
for t in `seq 0 6`;
do
	echo ${X[$t]}
	counter=$((t +440))
	echo $counter

	nohup srun --partition=phi --time=5- --mem=5000 python DQN.py $counter --exploration 1.0 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 6000000 --details "zombie , DQN" >logs/"$counter.out" &
done

cd action_memory
module load ffmpeg-3.2.2
counter=446
for j in `seq 1 5`;
do
X=(1111 4444 5423 6654 9111 1122 1337)
for t in `seq 0 6`;
do
	counter=$((counter +1))
	echo '#actions:'$j',seed:'${X[$t]},id:$counter,duel
	#nohup srun --partition=phi,long --time=5- --mem=3000 python duel_v2.py $counter --exploration 1.0 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 2000000 --details "zombie , duel_v2,nactions: $j" --naction $j >logs/"$counter.out" &
done
done

for j in `seq 1 5`;
do
X=(1111 4444 5423 6654 9111 1122 1337)
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#actions:'$j',seed:'${X[$t]},id:$counter,DQN
	nohup srun --partition=phi,long --time=5- --mem=3000 python DQN.py $counter --exploration 1.0 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 2000000 --details "zombie , duel_v2,nactions: $j" --naction $j >logs/"$counter.out" &
done
done

cd action_memory
module load ffmpeg-3.2.2
counter=502
for j in `seq 4 5`;
do
X=(1111 4444 5423 6654 9111 1122 1337)
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#actions:'$j',seed:'${X[$t]},id:$counter,DQN
	nohup srun --partition=phi,long,main --time=5- --mem=4000 python DQN.py $counter --exploration 1.0 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 6000000 --details "zombie , DQN,nactions: $j" --naction $j >logs/"$counter.out" &
done
done
#Competitive, Punishment with last 4 action.
cd action_memory
module load ffmpeg-3.2.2
counter=587
X=(1111 4444 5423 6654 9111 1122 1337)
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#actions:4,seed:'${X[$t]},id:$counter,Duel
	nohup srun --partition=phi,long,main --time=5- --mem=4000 python duel_v2.py $counter --exploration 0.1 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 500000 --details "Competitive, B:541 duel_v2,nactions: 4" --naction 4 --train_m 541 --target_m 541 --rwrdschem 0 1000 -0.1 >logs/"$counter.out" &
done

for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#actions:4,seed:'${X[$t]},id:$counter,Duel
	nohup srun --partition=phi,long,main --time=5- --mem=4000 python duel_v2.py $counter --exploration 0.1 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 500000 --details "Punishment, B:541 duel_v2,nactions: 4" --naction 4 --train_m 541 --target_m 541 --rwrdschem -10 1000 -0.1 >logs/"$counter.out" &
done

#DQN
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#actions:4,seed:'${X[$t]},id:$counter,DQN
	nohup srun --partition=phi,long,main --time=5- --mem=4000 python DQN.py $counter --exploration 0.1 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 500000 --details "Competitive, B:577 DQN,nactions: 4" --naction 4 --train_m 577 --target_m 577 --rwrdschem 0 1000 -0.1 >logs/"$counter.out" &
done

for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#actions:4,seed:'${X[$t]},id:$counter,DQN
	nohup srun --partition=phi,long,main --time=5- --mem=4000 python DQN.py $counter --exploration 0.1 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 500000 --details "Punishment, B:577 DQN,nactions: 4" --naction 4 --train_m 577 --target_m 577 --rwrdschem -10 1000 -0.1 >logs/"$counter.out" &
done
ACCOMPLISHED
#actions:4,seed:4444,id:504,DQN
#actions:4,seed:5423,id:505,DQN
#actions:4,seed:6654,id:506,DQN
#actions:4,seed:9111,id:507,DQN
#actions:4,seed:1122,id:508,DQN
#actions:4,seed:1337,id:509,DQN
#actions:5,seed:1111,id:510,DQN
#actions:5,seed:4444,id:511,DQN
#actions:5,seed:5423,id:512,DQN
#actions:5,seed:6654,id:513,DQN
#actions:5,seed:9111,id:514,DQN
#actions:5,seed:1122,id:515,DQN
#actions:5,seed:1337,id:516,DQN

#Competitive, Punishment with last 4 action.CR2
cd action_memory
module load ffmpeg-3.2.2
counter=615
X=(1111 4444 5423 6654 9111 1122 1337)
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#actions:4,seed:'${X[$t]},id:$counter,Duel
	nohup srun --partition=phi,long,main --time=5- --mem=4000 python duel_v2.py $counter --exploration 0.1 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 500000 --details "Competitive, B:541 duel_v2,nactions: 4,CR:2" --naction 4 --train_m 541 --target_m 541 --rwrdschem 0 1000 -0.1 >logs/"$counter.out" &
done

for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#actions:4,seed:'${X[$t]},id:$counter,Duel
	nohup srun --partition=phi,long,main --time=5- --mem=4000 python duel_v2.py $counter --exploration 0.1 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 500000 --details "Punishment, B:541 duel_v2,nactions: 4,CR:2" --naction 4 --train_m 541 --target_m 541 --rwrdschem -10 1000 -0.1 >logs/"$counter.out" &
done

#DQN
for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#actions:4,seed:'${X[$t]},id:$counter,DQN
	nohup srun --partition=phi,long,main --time=5- --mem=4000 python DQN.py $counter --exploration 0.1 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 500000 --details "Competitive, B:577 DQN,nactions: 4,CR:2" --naction 4 --train_m 577 --target_m 577 --rwrdschem 0 1000 -0.1 >logs/"$counter.out" &
done

for t in `seq 0 6`;
do
	counter=$((counter+1))
	echo '#actions:4,seed:'${X[$t]},id:$counter,DQN
	nohup srun --partition=phi,long,main --time=5- --mem=4000 python DQN.py $counter --exploration 0.1 --tau 0.001 --activation tanh --advantage max --seed ${X[$t]} --batch_size 32 --totalsteps 500000 --details "Punishment, B:577 DQN,nactions: 4,CR:2" --naction 4 --train_m 577 --target_m 577 --rwrdschem -10 1000 -0.1 >logs/"$counter.out" &
done
