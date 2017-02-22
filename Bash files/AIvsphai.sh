#### Phase 3, AI agent with zombie agent (nothing but the input of the other agent)#####
# trying the settings with the best 
:<<'Details'
Phase 3, AI agent with zombi agent (nothing but extra agent input for the network)
trying to check the hyperparameters with the new agent input.
Details
:<<'ACCOMPLISHED'
#cd APES
#counter=185
#nohup srun --partition=long --time=10- python duel.py $counter --exploration 0.01 --tau 0.001 --activation tanh --advantage max --seed 4917 --batch_size 32 --totalsteps 2000000 --details "zombie agent" >logs/"$counter.out" &

#cd APES
#counter=185
#nohup srun --partition=long --time=10- --mem=5000 python duel.py $counter --exploration 0.01 --tau 0.001 --activation tanh --advantage max --seed 4917 --batch_size 32 --totalsteps 1100000 --details "zombie agent" >logs/"$counter.out" &

#cd APES
#counter=186
#nohup srun --partition=long --time=10- --mem=5000 python duel.py $counter --exploration 0.01 --tau 0.001 --activation tanh --advantage max --seed 4917 --batch_size 32 --totalsteps 2000000 --details "zombie agent" >logs/"$counter.out" &

#FOR Video generation purpose PURPOSE ONLY
#cd APES
#counter=187
#nohup srun --partition=long --time=10- --mem=4000 python duel.py $counter --exploration 0.01 --tau 0.001 --activation tanh --advantage max --seed 4917 --batch_size 32 --totalsteps 1100000 --details "zombie agent" >logs/"$counter.out" &

cd APES2
counter=189
nohup srun --partition=long --time=10- --mem=5000 python duel.py $counter --exploration 0.01 --tau 0.001 --activation tanh --advantage max --seed 4917 --batch_size 32 --totalsteps 2000000 --details "zombie agent" >logs/"$counter.out" &

cd APES
counter=187
nohup srun --partition=long --time=10- --mem=8000 python duel.py $counter --exploration 0.01 --tau 0.001 --activation tanh --advantage max --seed 4917 --batch_size 32 --totalsteps 500000 --train_m 186 --target_m 186 --details "A*,180,FR,E186" >logs/"$counter.out" &

counter=188
nohup srun --partition=long --time=10- --mem=8000 python duel.py $counter --exploration 0.01 --tau 0.001 --activation tanh --advantage max --seed 4917 --batch_size 32 --totalsteps 1100000 --details "A*,180,FR,SCRCH" >logs/"$counter.out" &
ACCOMPLISHED

counter=189
cd APES
nohup srun --partition=long --time=10- python duel.py $counter --exploration 0.0 --tau 0.001 --activation tanh --advantage max --seed 4917 --batch_size 32 --totalsteps 5000 --train_m 187 --target_m 187 --details "A*,180,FR,E186" >logs/"$counter.out" &
