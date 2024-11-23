from mmdet3d.registry import DATASETS
from mmdet3d.datasets.det3d_dataset import Det3DDataset
from mmdet3d.structures import LiDARInstance3DBoxes
import numpy as np
import os.path as osp
import mmengine

@DATASETS.register_module()
class MyDataset(Det3DDataset):
    METAINFO = {
        'classes': ('LEP_metal', 'LEP_prom', 'vegetation')
    }

    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 modality=None,
                 default_cam_key='CAM2',
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 metainfo=None,
                 **kwargs):
        if metainfo is None:
            metainfo = self.METAINFO
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            modality=modality,
            default_cam_key=default_cam_key,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            metainfo=metainfo,
            **kwargs)
        self.num_ins_per_cat = [0] * len(self.metainfo['classes'])

    def load_data_list(self):
        """Loads the list of data from the annotation file."""
        print(f"Loading data from: {self.ann_file}")
        data_list = mmengine.load(self.ann_file)
        self.data_list = data_list  # Ensure data_list is assigned to self.data_list
        return data_list

    def get_data_info(self, index):
        """Gets and processes data information by index."""
        info = self.data_list[index]
        data_info = self.parse_data_info(info)
        return data_info

    def parse_data_info(self, info):
        """Processes raw data information."""
        data_info = dict()
        data_info['sample_idx'] = info.get('sample_idx', None)
        data_info['pts_filename'] = osp.join(
            self.data_prefix.get('pts', ''), info['point_cloud']['point_cloud_path'])
        data_info['ann_info'] = self.parse_ann_info(info)
        return data_info

    def parse_ann_info(self, info):
        """Processes annotations and returns ann_info."""
        annos = info.get('annos', None)
        if annos is None or len(annos['name']) == 0:
            ann_info = dict()
            ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros((0,), dtype=np.int64)
        else:
            names = annos['name']
            dims = np.array(annos['dimensions'])  # l, w, h
            locs = np.array(annos['location'])    # x, y, z
            rots = np.array(annos['rotation_y'])  # yaw

            # Form gt_bboxes_3d
            gt_bboxes_3d = np.concatenate([locs, dims, rots[:, np.newaxis]], axis=1)
            gt_labels_3d = np.array(
                [self.metainfo['classes'].index(n) for n in names], dtype=np.int64)
            ann_info = dict(
                gt_bboxes_3d=gt_bboxes_3d,
                gt_labels_3d=gt_labels_3d,
            )
            # Update instance counts
            for label in gt_labels_3d:
                self.num_ins_per_cat[label] += 1

        # Convert gt_bboxes_3d to LiDARInstance3DBoxes
        gt_bboxes_3d = LiDARInstance3DBoxes(ann_info['gt_bboxes_3d'])
        ann_info['gt_bboxes_3d'] = gt_bboxes_3d

        return ann_info
