from mmdet3d.registry import DATASETS
from mmdet3d.datasets.det3d_dataset import Det3DDataset
import numpy as np
import os.path as osp
import mmengine

@DATASETS.register_module()
class MyDataset(Det3DDataset):
    METAINFO = {
        'classes': ('LEP_prom', 'LEP_metal', 'vegetation'),
    }

    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=[],
                 modality=dict(use_lidar=True),
                 default_cam_key=None,
                 load_type='frame_based',
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 **kwargs):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            modality=modality,
            default_cam_key=default_cam_key,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)

    def load_data_list(self):
        data_list = mmengine.load(self.ann_file)
        for idx, data_info in enumerate(data_list):
            # Generate sample_id if not present
            if 'sample_id' not in data_info:
                point_cloud_path = data_info['point_cloud']['point_cloud_path']
                filename = osp.basename(point_cloud_path)
                sample_id = osp.splitext(filename)[0]
                data_info['sample_id'] = sample_id
        return data_list

    def parse_data_info(self, info):
        data_info = {}

        # Point cloud
        data_info['sample_idx'] = info['sample_id']
        data_info['pts_filename'] = osp.join(self.data_root, info['point_cloud']['point_cloud_path'])

        # Annotations
        if 'annos' in info:
            annos = info['annos']
            data_info['ann_info'] = self.parse_ann_info(annos)
        else:
            data_info['ann_info'] = None

        return data_info

    def parse_ann_info(self, ann_info):
        gt_bboxes_3d = []
        gt_labels_3d = []

        names = ann_info['name']
        locations = ann_info['location']
        dimensions = ann_info['dimensions']
        rotations = ann_info['rotation_y']

        for name, loc, dim, rot in zip(names, locations, dimensions, rotations):
            # Convert to numpy arrays
            loc = np.array(loc, dtype=np.float32)
            dim = np.array(dim, dtype=np.float32)
            rot = float(rot)

            # Adjust dimensions if necessary (e.g., from hwl to lwh)
            # Here, assuming dimensions are [l, w, h]
            l, w, h = dim  # Adjust order if necessary

            # Create bounding box [x, y, z, dx, dy, dz, yaw]
            bbox = [loc[0], loc[1], loc[2], l, w, h, rot]
            gt_bboxes_3d.append(bbox)

            # Map class name to label index
            label = self.metainfo['classes'].index(name)
            gt_labels_3d.append(label)

        ann = {
            'gt_bboxes_3d': np.array(gt_bboxes_3d, dtype=np.float32),
            'gt_labels_3d': np.array(gt_labels_3d, dtype=np.int64)
        }

        return ann
