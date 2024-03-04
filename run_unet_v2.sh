GPU_ID=$1
DIR=$2
gpu_list=(0 4 5 6 7)
idx=0

SEG_DIR="fss_comparison/${DIR}"
SAVE_DIR="orig_512_seg/${DIR}"
SNAP_DIR="output/${DIR}"

for obj in "breakfast_box" "screw_bag" "juice_bottle" "splicing_connectors" "pushpins"
do
    CUDA_VISIBLE_DEVICES=${gpu_list[$idx]} python train_normal_unet.py \
    --obj_name ${obj} \
    --num_epochs 300 \
    --snapshot_dir ${SAVE_DIR} \
    --save_dir ${SAVE_DIR} \
    --seg_dir ${SEG_DIR} \
    --learning_rate 1e-3 \
    --pretrained False &

    idx=$(($idx+1))
    sleep 1
done

exit 0