from mmdet3d.registry import DATASETS
from mmdet3d.datasets import Custom3DDataset
from mmdet3d.structures import LiDARInstance3DBoxes
import numpy as np
import os.path as osp

@DATASETS.register_module()
class MyDataset(Custom3DDataset):

    METAINFO = {
        'classes': ('LEP_metal', 'LEP_prom', 'vegetation')
    }

    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 modality=None,
                 filter_empty_gt=True,
                 test_mode=False):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)
        self.data_root = data_root

    def load_data_list(self):
        """Загружает список данных из файла аннотаций."""
        data_list = mmengine.load(osp.join(self.data_root, self.ann_file))
        return data_list

    def get_data_info(self, index):
        info = self.data_list[index]
        input_dict = dict(
            sample_idx=index,
            pts_filename=osp.join(self.data_root, info['point_cloud']['point_cloud_path']),
            timestamp=info.get('timestamp', 0),
        )
        return input_dict

    def get_ann_info(self, index):
        info = self.data_list[index]
        annos = info.get('annos', None)
        if annos is None or len(annos['name']) == 0:
            gt_bboxes_3d = np.zeros((0, 7), dtype=np.float32)
            gt_labels_3d = np.zeros((0, ), dtype=np.int64)
        else:
            names = annos['name']
            dims = np.array(annos['dimensions'])
            locs = np.array(annos['location'])
            rots = np.array(annos['rotation_y'])
            gt_bboxes_3d = np.concatenate([locs, dims, rots[:, np.newaxis]], axis=1)
            gt_labels_3d = np.array([self.metainfo['classes'].index(n) for n in names], dtype=np.int64)
        gt_bboxes_3d = LiDARInstance3DBoxes(gt_bboxes_3d)
        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
        )
        return anns_results
