#!/bin/bash -x

#   KNL run     #
# On deepv:  srun --partition=knl -N 1 -n 1 --cpus-per-task=64 --pty /bin/bash -i
# 64 cores compares to AIBT with 68
# Keep 100 of warmup otherwise fps growing over time

cd /sdv-work/lmontigny/0_Intel/AIPG/benchmarks-master
module load cmake/3.6.2 gcc/7.2.0-64bit anaconda3/5.0.0.1 intel/18.1
source activate tf_compiled


export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
export OMP_NUM_THREADS=64

# Google net, batch 128
numactl --preferred 1 python Jenkins/TensorFlow/benchmarks/mkl_tf_cnn_benchmarks.py  --mkl=True --forward_only=False --data_name=imagenet --num_batches=100 --weight_decay=4e-05 --final_evaluation=False --kmp_blocktime=1 --save_model_steps=6000 --data_dir=/sdv-work/lmontigny/0_Intel/AIPG/1G_imagenet_dataset/ --num_inter_threads=3 --evaluate_every=6000 --distortions=True --optimizer=sgd --batch_size=128 --device=cpu --num_warmup_batches=100 --num_intra_threads=64 --local_parameter_device=cpu --display_every=10 --data_format=NCHW --model=googlenet --cpu=knl

# Resnet 50 v1, batch size 128
export KMP_BLOCKTIME=0
numactl --preferred 1 python Jenkins/TensorFlow/benchmarks/mkl_tf_cnn_benchmarks.py  --mkl=True --forward_only=False --data_name=imagenet --num_batches=100 --final_evaluation=False --kmp_blocktime=0 --save_model_steps=18000 --data_dir=/sdv-work/lmontigny/0_Intel/AIPG/1G_imagenet_dataset/ --num_inter_threads=3 --evaluate_every=18000 --distortions=True --optimizer=sgd --batch_size=128 --device=cpu --num_warmup_batches=100 --num_intra_threads=64 --local_parameter_device=cpu --display_every=10 --data_format=NCHW --model=resnet50 --cpu=knl

# Inception v3, batch 96
export KMP_BLOCKTIME=0
numactl --preferred 1 python Jenkins/TensorFlow/benchmarks/mkl_tf_cnn_benchmarks.py  --mkl=True --forward_only=False --data_name=imagenet --num_batches=100 --final_evaluation=False --kmp_blocktime=0 --save_model_steps=20000 --data_dir=/sdv-work/lmontigny/0_Intel/AIPG/1G_imagenet_dataset/ --num_inter_threads=3 --evaluate_every=20000 --distortions=True --optimizer=sgd --batch_size=96 --device=cpu --num_warmup_batches=100 --num_intra_threads=64 --local_parameter_device=cpu --display_every=10 --data_format=NCHW --model=inception3 --cpu=knl

