source /miniconda/etc/profile.d/conda.sh
conda init bash
conda activate core
export PYTHONPATH=.:$PYTHONPATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin:/usr/lib/nvidia-000
tensorboard --logdir ${BOLT_ARTIFACT_DIR} --bind_all --port ${TENSORBOARD_PORT} &
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python train.py
