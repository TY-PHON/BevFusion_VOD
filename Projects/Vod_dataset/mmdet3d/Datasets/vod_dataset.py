import mmcv
from typing import Any, Dict
import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets.custom_3d import Custom3DDataset
import os.path as osp
import tempfile

@DATASETS.register_module()
class VodDataset(Custom3DDataset):
    NameMapping = {

    }

    ErrNameMapping = {

    }

    CLASSES = (
        "Car",
        "Pedestrian",
        "Cyclist",
        "rider",
        "bicycle",
    )

    def __init__(
        self,
        ann_file,
        pipeline=None,
        dataset_root=None,
        object_classes=None,
        load_interval=1,
        with_velocity=False,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        data_config=None,
        test_mode=False,
        use_valid_flag=False,
    ) -> None:
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        super().__init__(
            dataset_root=dataset_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=object_classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
        )
        ##self.map_classes = map_classes

        self.with_velocity = with_velocity
        self.data_config = data_config
        ##from nuscenes.eval.detection.config import config_factory

        ##self.eval_detection_configs = config_factory(self.eval_version)
        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )
    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info["valid_flag"]
            gt_names = set(info["gt_names"][mask])
        else:
            gt_names = set(info["gt_names"])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids
    
    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data["infos"], key=lambda e: e["frameId"]))
        data_infos = data_infos[:: self.load_interval]
        self.metadata = data["metadata"]
        self.version = self.metadata["version"]
        return data_infos
    
    def get_data_info(self, index: int) -> Dict[str, Any]:
        info = self.data_infos[index]

        data = dict(
            ##token=info["token"],
            ##sample_idx=info['token'],
            frame_id=info["frameId"],
            lidar_path=info["lidarPath"],
            image_paths=info["camPath"],
            sweeps=info["sweeps"]
            ##location=info["location"],
        )
        
        # ego to global transform
        # ego2global = np.eye(4).astype(np.float32)
        # ego2global[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
        # ego2global[:3, 3] = info["ego2global_translation"]
        # data["ego2global"] = ego2global

        # lidar to ego transform
        # lidar2ego = np.eye(4).astype(np.float32)
        # lidar2ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        # lidar2ego[:3, 3] = info["lidar2ego_translation"]
        # data["lidar2ego"] = lidar2ego

        #暂时没加
        if self.modality["use_camera"]:
            ##data["image_paths"] = []
            data["lidar2camera"] = []
            # data["lidar2image"] = []
            # data["camera2ego"] = []
            data["camera_intrinsics"] = []
            data["camera2lidar"] = []

            for _, camera_info in info["cams"].items():
                
                # data["image_paths"].append(camera_info["camPath"])

                # lidar to camera transform
                lidar2camera_rt = np.eye(4).astype(np.float64)
                lidar2camera_rt[:3, :4] = camera_info['lidar2camera']
                data["lidar2camera"].append(lidar2camera_rt)

                # camera to lidar transform
                camera2lidar = np.array(np.linalg.inv(lidar2camera_rt), dtype=np.float64)
                data["camera2lidar"].append(camera2lidar)

                # camera intrinsics
                camera_intrinsics = np.eye(4).astype(np.float32)
                camera_intrinsics[:3, :4] = camera_info["camera_intrinsics"]
                data["camera_intrinsics"].append(camera_intrinsics)

                # # lidar to image transform
                # lidar2image = camera_intrinsics @ lidar2camera_rt.T
                # data["lidar2image"].append(lidar2image)

                # # camera to ego transform
                # camera2ego = np.eye(4).astype(np.float32)
                # camera2ego[:3, :3] = Quaternion(
                #     camera_info["sensor2ego_rotation"]
                # ).rotation_matrix
                # camera2ego[:3, 3] = camera_info["sensor2ego_translation"]
                # data["camera2ego"].append(camera2ego)

                

        annos = self.get_ann_info(index)
        data["ann_info"] = annos
        ##print('image_path_in_vodDataset:', data)
        return data
    
    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        # filter out bbox containing no points
        ## but our dataset don't have this

        # if self.use_valid_flag:
        #     mask = info["valid_flag"]
        # else:
        #     mask = info["num_lidar_pts"] > 0
        gt_bboxes_3d = info["gt_boxes"]
        gt_names_3d = info["gt_names"]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        ##TODO filter
        ##our dataset don't have this
        # if self.with_velocity:
        #     gt_velocity = info["gt_velocity"]
        #     nan_mask = np.isnan(gt_velocity[:, 0])
        #     gt_velocity[nan_mask] = [0.0, 0.0]
        #     gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        # haotian: this is an important change: from 0.5, 0.5, 0.5 -> 0.5, 0.5, 0
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0)
        ).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
        )
        return anns_results
    
    ##TODO evaluate MAP
        