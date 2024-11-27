from mmdet3d.registry import DATASETS
from mmdet3d.datasets.det3d_dataset import Det3DDataset
import numpy as np
import os.path as osp
import mmengine

@DATASETS.register_module()
class MyDataset(Det3DDataset):
    METAINFO = {
        'classes': ('LEP_prom', 'LEP_metal', 'vegetation')
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
        """
        Загружает список данных и добавляет `sample_id` при необходимости.
        """
        data_dict = mmengine.load(self.ann_file)
        data_list = data_dict['data_list']
        metainfo = data_dict['metainfo']
        self.metainfo = metainfo  # Обновляем метаинформацию

        for idx, data_info in enumerate(data_list):
            # Проверка наличия `sample_id`
            if 'sample_id' not in data_info:
                point_cloud_path = data_info['point_cloud']['point_cloud_path']
                filename = osp.basename(point_cloud_path)
                sample_id = osp.splitext(filename)[0]
                data_info['sample_id'] = sample_id
            # Дополнительные проверки можно добавить здесь
        return data_list

    def parse_data_info(self, info):
        """
        Обрабатывает информацию о данных и аннотациях.
        """
        data_info = {}

        # Идентификатор выборки
        data_info['sample_idx'] = info['sample_id']
        data_info['pts_filename'] = osp.join(self.data_root, info['point_cloud']['point_cloud_path'])

        # Аннотации
        if 'annos' in info and info['annos']['name']:
            annos = info['annos']
            parsed_annos = self.parse_ann_info(annos)
            if parsed_annos is not None:
                data_info['ann_info'] = parsed_annos
        else:
            if not self.test_mode:
                # Для тренировочных и валидационных данных, аннотации обязательны
                data_info['ann_info'] = None
            else:
                # Для тестовых данных можно оставить аннотации пустыми
                data_info['ann_info'] = {
                    'gt_bboxes_3d': np.zeros((0, 7), dtype=np.float32),
                    'gt_labels_3d': np.array([], dtype=np.int64)
                }

        return data_info

    def parse_ann_info(self, ann_info):
        """
        Обрабатывает аннотации и преобразует их в формат, ожидаемый моделью.
        """
        gt_bboxes_3d = []
        gt_labels_3d = []

        names = ann_info['name']
        locations = ann_info['location']
        dimensions = ann_info['dimensions']
        rotations = ann_info['rotation_y']

        for name, loc, dim, rot in zip(names, locations, dimensions, rotations):
            # Преобразование в numpy массивы
            loc = np.array(loc, dtype=np.float32)
            dim = np.array(dim, dtype=np.float32)
            rot = float(rot)

            # Предполагаем, что размеры [l, w, h]
            l, w, h = dim

            # Создание бокса [x, y, z, dx, dy, dz, yaw]
            bbox = [loc[0], loc[1], loc[2], l, w, h, rot]
            gt_bboxes_3d.append(bbox)

            # Преобразование имени класса в индекс
            try:
                label = self.metainfo['classes'].index(name)
                gt_labels_3d.append(label)
            except ValueError:
                print(f"Неизвестный класс: {name}. Пропуск.")
                continue

        if not gt_bboxes_3d:
            if not self.test_mode:
                # Для тренировочных и валидационных данных, если нет аннотаций, возвращаем None
                return None
            else:
                return {
                    'gt_bboxes_3d': np.zeros((0, 7), dtype=np.float32),
                    'gt_labels_3d': np.array([], dtype=np.int64)
                }

        ann = {
            'gt_bboxes_3d': np.array(gt_bboxes_3d, dtype=np.float32),
            'gt_labels_3d': np.array(gt_labels_3d, dtype=np.int64)
        }

        print(f"Обработано {len(gt_bboxes_3d)} аннотаций для sample_id {info.get('sample_id', 'unknown')}.")
        return ann
