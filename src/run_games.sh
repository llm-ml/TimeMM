export CUDA_VISIBLE_DEVICES=1

source activate mmrec

torchrun --nproc_per_node=1 main.py -m TIMEMM -d games