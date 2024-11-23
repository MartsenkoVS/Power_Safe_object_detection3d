from mmdet3d.registry import DATASETS
from mmdet3d.datasets import Det3DDataset
from mmdet3d.structures import LiDARInstance3DBoxes
import numpy as np
import mmengine

@DATASETS.register_module()
class MyDataset(Det3DDataset):

    METAINFO = {
        'classes': ('LEP_metal', 'LEP_prom', 'vegetation')
    }

    def load_data_list(self):
        """Загружает список данных из файла аннотаций."""
        if isinstance(self.ann_file, str):
            data_list = mmengine.load(self.ann_file)
        else:
            data_list = self.ann_file
        return data_list

    def parse_ann_info(self, info):
        """Обрабатывает `instances` в `info` и возвращает `ann_info`."""
        ann_info = super().parse_ann_info(info)
        if ann_info is None:
            ann_info = dict()
            # Пустой экземпляр
            ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)

        # Фильтруем классы, не используемые в обучении
        ann_info = self._remove_dontcare(ann_info)
        gt_bboxes_3d = LiDARInstance3DBoxes(ann_info['gt_bboxes_3d'])
        ann_info['gt_bboxes_3d'] = gt_bboxes_3d
        return ann_info
