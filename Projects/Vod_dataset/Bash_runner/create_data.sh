ROOT_PATH_PROJ='/home/user/pzc/bevfusion-main/Projects'
ROOT_PATH_DATASET=${ROOT_PATH_DATASET}'Vod_dataset/view_of_delft_PUBLIC/lidar'
echo ${ROOT_PATH_DATASET}
python Tools/create_data.py Vod --root-path ${ROOT_PATH_DATASET} --out-dir ${ROOT_PATH_DATASET} --extra-tag Vod
