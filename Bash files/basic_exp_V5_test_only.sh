module load ffmpeg-3.2.2
cd APES
nohup srun --partition=main --time=6- --mem=4000 python duel_V5_test_different_exploration.py 5 --exploration 0.5 --episodes 10000 --naction 4 --svision 360 --max_timesteps 100 --rwrdschem 0 1000 -0.1 >logs/1.out &
nohup srun --partition=main --time=6- --mem=4000 python duel_V5_test_different_exploration.py 6 --exploration 0.1 --episodes 10000 --naction 4 --svision 360 --max_timesteps 100 --rwrdschem 0 1000 -0.1 >logs/1.out &
nohup srun --partition=main --time=6- --mem=4000 python duel_V5_test_different_exploration.py 7 --exploration 0.05 --episodes 10000 --naction 4 --svision 360 --max_timesteps 100 --rwrdschem 0 1000 -0.1 >logs/1.out &
nohup srun --partition=main --time=6- --mem=4000 python duel_V5_test_different_exploration.py 8 --exploration 0.0 --episodes 10000 --naction 4 --svision 360 --max_timesteps 100 --rwrdschem 0 1000 -0.1 >logs/1.out &
