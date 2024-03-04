GPU_ID=$1
DIR=$2
mtype=$3
stype=$4
gpu_list=(0 4 5 6 7)
idx=0

SEG_DIR="orig_512_seg/${DIR}"
CUDA_VISIBLE_DEVICES=${GPU_ID} python psad.py \
    --type logical \
    --standardize 1 \
    --avgpool_size 5 \
    --less_data 1 \
    --memory_type ${mtype} \
    --scale_type ${stype} \
    --seg_dir ${SEG_DIR} \
    --save_img 0 \
    --save_csv 0

CUDA_VISIBLE_DEVICES=${GPU_ID} python psad.py \
    --type structural \
    --standardize 1 \
    --avgpool_size 5 \
    --less_data 1 \
    --memory_type ${mtype} \
    --scale_type ${stype} \
    --seg_dir ${SEG_DIR} \
    --save_img 0 \
    --save_csv 0

exit 0


