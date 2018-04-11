#!/bin/bash -x

#   SKX run     #
# Keep 100 of warmup otherwise fps growing over time

cd /sdv-work/lmontigny/0_Intel/AIPG/benchmarks-master
module load cmake/3.6.2 gcc/7.2.0-64bit anaconda3/5.0.0.1 intel/18.1
source activate tf_compiled


export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
export OMP_NUM_THREADS=56

# Google net, batch 128
python Jenkins/TensorFlow/benchmarks/mkl_tf_cnn_benchmarks.py  --mkl=True --forward_only=False --data_name=imagenet --num_batches=100 --weight_decay=4e-05 --final_evaluation=False --kmp_blocktime=1 --save_model_steps=6000 --data_dir=/home/lmontigny/0_DEEP_EST/AIPG/data/1G_imagenet_dataset/ --num_inter_threads=2 --evaluate_every=6000 --distortions=True --optimizer=sgd --batch_size=128 --device=cpu --num_warmup_batches=100 --num_intra_threads=56 --local_parameter_device=cpu --display_every=10 --data_format=NCHW --model=googlenet --cpu=skx

# Resnet 50 v1, batch size 128
export KMP_BLOCKTIME=0
python Jenkins/TensorFlow/benchmarks/mkl_tf_cnn_benchmarks.py  --mkl=True --forward_only=False --data_name=imagenet --num_batches=100 --final_evaluation=False --kmp_blocktime=0 --save_model_steps=18000 --data_dir=/home/lmontigny/0_DEEP_EST/AIPG/data/1G_imagenet_dataset/ --num_inter_threads=2 --evaluate_every=18000 --distortions=True --optimizer=sgd --batch_size=128 --device=cpu --num_warmup_batches=100 --num_intra_threads=56 --local_parameter_device=cpu --display_every=10 --data_format=NCHW --model=resnet50 --cpu=skx

# Inception v3, batch 96
export KMP_BLOCKTIME=0
python Jenkins/TensorFlow/benchmarks/mkl_tf_cnn_benchmarks.py  --mkl=True --forward_only=False --data_name=imagenet --num_batches=100 --final_evaluation=False --kmp_blocktime=0 --save_model_steps=20000 --data_dir=/home/lmontigny/0_DEEP_EST/AIPG/data/1G_imagenet_dataset/ --num_inter_threads=2 --evaluate_every=20000 --distortions=True --optimizer=sgd --batch_size=96 --device=cpu --num_warmup_batches=100 --num_intra_threads=56 --local_parameter_device=cpu --display_every=10 --data_format=NCHW --model=inception3 --cpu=skx
