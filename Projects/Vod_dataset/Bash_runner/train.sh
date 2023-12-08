DATE=$(date '+%Y-%m-%d_%H-%M-%S')
TRAIN_PY='Tools/train.py'
CONFIG_FILE='Vod_dataset/configs/bevfusion_c_l_meg.py'
WORK_DIR="runs/${DATE}/"

torchpack dist-run -np 1 python ${TRAIN_PY} ${CONFIG_FILE} --run-dir ${WORK_DIR}