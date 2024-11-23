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
        """Загружает список данных из файла аннотаций."""
        data_list = mmengine.load(osp.join(self.data_root, self.ann_file))
        return data_list

    def parse_data_info(self, info):
        """Обрабатывает исходную информацию о данных."""
        data_info = dict()
        data_info['sample_idx'] = info.get('sample_idx', None)
        data_info['pts_filename'] = osp.join(self.data_root, info['point_cloud']['point_cloud_path'])
        data_info['ann_info'] = self.parse_ann_info(info)
        return data_info

    def parse_ann_info(self, info):
        """Обрабатывает аннотации и возвращает ann_info."""
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

            # Формируем gt_bboxes_3d
            gt_bboxes_3d = np.concatenate([locs, dims, rots[:, np.newaxis]], axis=1)
            gt_labels_3d = np.array(
                [self.metainfo['classes'].index(n) for n in names], dtype=np.int64)
            ann_info = dict(
                gt_bboxes_3d=gt_bboxes_3d,
                gt_labels_3d=gt_labels_3d,
            )

        # Преобразуем gt_bboxes_3d в LiDARInstance3DBoxes
        gt_bboxes_3d = LiDARInstance3DBoxes(ann_info['gt_bboxes_3d'])
        ann_info['gt_bboxes_3d'] = gt_bboxes_3d

        return ann_info
