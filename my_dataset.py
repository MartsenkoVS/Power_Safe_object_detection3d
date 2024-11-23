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
                 filter_empty_gt=True,  # Временно установите на False для отладки
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
        print(f"metainfo: {self.metainfo}")

    def load_data_list(self):
        """Loads the list of data from the annotation file."""
        print(f"Loading data from: {self.ann_file}")
        data_list = mmengine.load(self.ann_file)
        self.data_list = data_list  # Ensure data_list is assigned to self.data_list
        print(f"Loaded {len(self.data_list)} items in data_list")
        if len(self.data_list) > 0:
            print(f"Первый элемент в data_list: {self.data_list[0]}")
        return data_list

    def get_data_info(self, index):
        """Gets and processes data information by index."""
        print(f"Доступ к индексу: {index}")
        if index >= len(self.data_list) or index < 0:
            print(f"Индекс {index} выходит за пределы. Длина data_list: {len(self.data_list)}")
            raise IndexError(f"Индекс {index} выходит за границы для data_list длиной {len(self.data_list)}")
        info = self.data_list[index]
        data_info = self.parse_data_info(info)
        print(f"Processed ann_info for index {index}")
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
        print("Вызов parse_ann_info")
        annos = info.get('annos', None)
        if annos is None:
            print("Нет ключа 'annos' в информации о данных.")
        else:
            print(f"Ключи аннотаций: {annos.keys()}")

        if annos is None or len(annos.get('name', [])) == 0:
            ann_info = dict()
            ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros((0,), dtype=np.int64)
            print("Аннотации пусты.")
        else:
            names = annos['name']
            dims = np.array(annos['dimensions'])  # l, w, h
            locs = np.array(annos['location'])    # x, y, z
            rots = np.array(annos['rotation_y'])  # yaw

            print(f"Обработка {len(names)} аннотаций.")

            try:
                gt_labels_3d = np.array(
                    [self.metainfo['classes'].index(n) for n in names], dtype=np.int64)
                print(f"gt_labels_3d: {gt_labels_3d}")
            except ValueError as e:
                print(f"Ошибка при индексации меток: {e}")
                gt_labels_3d = np.zeros((0,), dtype=np.int64)

            # Проверка размеров массивов
            if dims.ndim == 1:
                dims = dims.reshape(1, -1)
            if locs.ndim == 1:
                locs = locs.reshape(1, -1)
            if rots.ndim == 1:
                rots = rots.reshape(1, -1)

            gt_bboxes_3d = np.concatenate([locs, dims, rots[:, np.newaxis]], axis=1)
            ann_info = dict(
                gt_bboxes_3d=gt_bboxes_3d,
                gt_labels_3d=gt_labels_3d,
            )
            # Обновление счётчиков экземпляров
            for label in gt_labels_3d:
                if 0 <= label < len(self.num_ins_per_cat):
                    self.num_ins_per_cat[label] += 1
                else:
                    print(f"Некорректная метка {label} для классов {self.metainfo['classes']}")

            print(f"Текущие счётчики экземпляров: {self.num_ins_per_cat}")

        # Преобразуем gt_bboxes_3d в LiDARInstance3DBoxes
        if 'gt_bboxes_3d' in ann_info:
            gt_bboxes_3d = LiDARInstance3DBoxes(ann_info['gt_bboxes_3d'])
            ann_info['gt_bboxes_3d'] = gt_bboxes_3d
        else:
            ann_info['gt_bboxes_3d'] = LiDARInstance3DBoxes(np.zeros((0, 7), dtype=np.float32))

        return ann_info
