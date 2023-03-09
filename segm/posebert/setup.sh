#!/bin/bash
echo "Use CUDA 11.0"
export CUDA_HOME=/nfs/core/cuda/11.0/
export LD_LIBRARY_PATH=/nfs/core/cuda/11.0/lib64:cuda-11.0-cudnn-7.6.5.32/lib64:${LD_LIBRARY_PATH}
export CUDA_SAMPLES_INC=/nfs/core/cuda/11.0/samples/common/inc/

echo "Activating g++ 7"
source scl_source enable devtoolset-7
export CXX=g++

echo "Activating conda env"
source /nfs/tools/humans/conda/bin/activate posebert
export LD_LIBRARY_PATH=/nfs/tools/humans/conda/envs/posebert/lib:$LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

echo "CUDA info:"
nvidia-smi
nvidia-smi -L
env | grep CUDA

echo "PYTHONPATH"
export PYTHONPATH=$PYTHONPATH:`pwd`
echo "PYTHONPATH=${PYTHONPATH}"

echo "OpenGL"
export EGL_DEVICE_ID=$( nvidia-container-cli info | grep Minor | awk '{print $3}' | head -n 1)
export PYOPENGL_PLATFORM=egl
echo "EGL_DEVICE_ID=${EGL_DEVICE_ID}"