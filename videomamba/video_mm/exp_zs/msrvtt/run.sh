export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

JOB_NAME='m16_5m'
OUTPUT_DIR="exp_zs/msrvtt/$JOB_NAME"
LOG_DIR="exp_zs/msrvtt/logs/${JOB_NAME}"
PARTITION='LocalQ' #'video5'
NNODE=1
NUM_GPUS=1
NUM_CPU=1

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    -n${NNODE} \
    --gres=gpu:${NUM_GPUS} \
    --ntasks-per-node=1 \
    --cpus-per-task=${NUM_CPU} \
    bash torchrun.sh \
    --nnodes=${NNODE} \
    --nproc_per_node=${NUM_GPUS} \
    --rdzv_backend=c10d \
    tasks/retrieval.py \
    exp_zs/msrvtt/config.py \
    output_dir ${OUTPUT_DIR} \
    evaluate True \
    zero_shot True \
    pretrained_path /home/tony/videomamba_m16_5M_f8_res224.pth #your_model_path/videomamba_m16_25M_f8_res224.pth     evaluate True \


