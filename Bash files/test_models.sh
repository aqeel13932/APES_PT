cd APES2 
srun --partition=main --time=2- --cpus=4 --mem=8000 python test_models.py --naction 4 --models 939 9{4,5,6,7}{0,1,2,3,4,5,6,7,8,9} 98{0,1,2,3,4,5,6,7} > test_models.out & 
