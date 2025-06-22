source activate base


conda create -n reward_bench python=3.10 -y

source activate reward_bench

pip install -e ".[generative]"