import os
from collections import OrderedDict
from os import path as osp
from typing import List, Tuple, Union

import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box

from mmdet3d.core.bbox.box_np_ops import points_cam2img
from mmdet3d.datasets import NuScenesDataset

import math

vod_categories = (
    "Car",
    "Pedestrian",
    "Cyclist",
    "rider",
    "bicycle",
    "bicycle_rack",
    "human_depiction",

)

### take apart train val
def getTrainValScenes(rootPath):
    s = osp.join(rootPath, 'training', 'velodyne')
    trainData = osp.join(rootPath, 'ImageSets', 'train.txt')
    valData = osp.join(rootPath, 'ImageSets', 'val.txt')

    print('s:', s)

    # all_scenes = os.listdir(s)
    
    # all_scenes = [scenes.split('.')[0] for scenes in all_scenes]
    # all_scenes.sort(key=lambda x: int(x.split("/")[-1]))
    

    # testNum = math.floor(len(all_scenes) / 10)
    # trainNum = len(all_scenes) - testNum
    # trainScenes = all_scenes[:trainNum]
    # valScenes = all_scenes[trainNum:]
    ##print('valScenes:', valScenes)

    trainScenes = []
    with open(trainData, 'r') as f:
        for scene in f.readlines():
            scene = scene.strip('\n')
            trainScenes.append(scene)
    
    valScenes = []
    with open(valData, 'r') as f:
        for scene in f.readlines():
            scene = scene.strip('\n')
            valScenes.append(scene)

    return trainScenes, valScenes


def createVodInfos(rootPath, info_prefix):
    #info_prefix implaments extra_tag which is Vod
    trainScenes, valScenes = getTrainValScenes(rootPath)
    
    trainVodInfos, valVodInfos = fillTrainValInfos(rootPath, trainScenes, valScenes)

    metadata = dict(version="v1.0-mini")

    print('train nums:{}, val nums:{}'.format(len(trainVodInfos), len(valVodInfos)))

    data = dict(infos=trainVodInfos, metadata=metadata)
    info_path = osp.join(rootPath, "{}_infos_train.pkl".format(info_prefix))
    mmcv.dump(data, info_path)
    data["infos"] = valVodInfos
    info_val_path = osp.join(rootPath, "{}_infos_val.pkl".format(info_prefix))
    mmcv.dump(data, info_val_path)

### rootPath = view_of_delft_PUBLIC\lidar
### trainScenes = 00001,00002......
def fillTrainValInfos(rootPath, trainScenes, valScenes, test=False):
    
    
    rootPath = osp.join(rootPath, 'training')
    print('rootPath:', rootPath)
    trainVodInfos = []
    valVodInfos = []

    sceneNames = trainScenes + valScenes
    ##print('trainPath:', trainScenes)
    ##sceneNames = osp.join(sceneNames, '.bin')
    lidarPath = osp.join(rootPath, 'velodyne')
    camPath = osp.join(rootPath, 'image_2')
    labelPath = osp.join(rootPath, 'label_2')
    calibPath = osp.join(rootPath, 'calib')
    frameId = ''
    for sid, frameId in enumerate(sceneNames):
        #print('frameId path:', frameId) 

        
        camPath_for = osp.join(camPath, frameId+'.jpg')
        lidarPath_for = osp.join(lidarPath, frameId+'.bin')
        labelPath_for = osp.join(labelPath, frameId+'.txt')
        calibPath_for = osp.join(calibPath, frameId+'.txt')
            
        ##print('labelPath:', labelPath_for)

        ##binData = np.fromfile(lidarPath, dtype=np.float32).reshape(-1,4)

        #timeStemp
        info = {
            "frameId": frameId,
            "lidarPath": lidarPath_for,
            "camPath": camPath_for,
            "sweeps": [],
            "cams": dict()
        }
        print('camPath:', info['camPath'])
        lidar2cam = []
        cam_intrinsics = []
        with open(calibPath_for, 'r') as f:
            for calInfo in f.readlines():
                calInfo = calInfo.strip('\n')

                calList = calInfo.split()

                if calList[0] == 'Tr_velo_to_cam:':
                    lidar2cam = np.array(calList[1:]).reshape(3, 4)
                    lidar2cam = lidar2cam.astype('float64')
                elif calList[0] == 'P0:':
                    cam_intrinsics = np.array(calList[1:]).reshape(3, 4)
                    cam_intrinsics = cam_intrinsics.astype('float64')

        # print('cam_intri:', cam_intrinsics)
        # print('lidar2cam:', lidar2cam)
        cam_info = dict(
            camPath=camPath_for,
            lidar2camera=lidar2cam,
            camera_intrinsics=cam_intrinsics
        )
        info["cams"].update({'cam': cam_info})
        ##annotation
        if not test:
            labels = []
            with open(labelPath_for, 'r') as f:
                for label in f.readlines():
                    label = label.strip('\n')
                    
                    labelList = label.split() 
                    #print('label[8:11]:', labelList[8:11])
                    labels.append(labelList)

                locs = np.array([[
                    label[11:14]]
                    for label in labels         
                ]).reshape(-1, 3)
                locs = locs.astype('float64')                 
                dims = np.array([[
                    label[8:11]]
                    for label in labels
                ]).reshape(-1, 3)
                dims = dims.astype('float64')
                rots = np.array([[
                    label[14]]
                    for label in labels
                ]).reshape(-1, 1)
                rots = rots.astype('float64')
                names = np.array([label[0] for label in labels])
                ##print('ann_names:', names)

                ##convert rot to SECOND format
                gt_boxes = np.concatenate([locs, dims, -(rots+np.pi/2)], axis=1)
                assert len(gt_boxes) == len(labels), f"{len(gt_boxes)}, {len(labels)}"
                info["gt_boxes"] = gt_boxes
                info["gt_names"] = names

        if frameId in trainScenes:
            trainVodInfos.append(info)
        elif frameId in valScenes:
            valVodInfos.append(info)

    return trainVodInfos, valVodInfos


    
