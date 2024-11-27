from mmdet3d.registry import DATASETS
from mmdet3d.datasets.det3d_dataset import Det3DDataset
import numpy as np
import os.path as osp
import mmengine
import logging

@DATASETS.register_module()
class MyDataset(Det3DDataset):
    METAINFO = {
        'classes': ('LEP_prom', 'LEP_metal', 'vegetation'),
        # 'palette': [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Удалено, так как не требуется
    }

    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=[],
                 modality=dict(use_lidar=True),
                 default_cam_key=None,
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
        # Дополнительная инициализация, если необходимо

    def load_data_list(self):
        """
        Загружает список данных из файла аннотаций и обновляет `sample_id`, если его нет.
        """
        data_dict = mmengine.load(self.ann_file)
        
        # Проверяем, что файл содержит необходимые ключи
        if 'metainfo' not in data_dict or 'data_list' not in data_dict:
            raise ValueError("Файл аннотаций должен содержать ключи 'metainfo' и 'data_list'")
        
        metainfo = data_dict['metainfo']
        raw_data_list = data_dict['data_list']

        # Обновляем метаинформацию через внутренний атрибут
        self._metainfo = self._load_metainfo(metainfo)
        logging.info(f"Метаинформация обновлена: {self._metainfo}")
        
        data_list = raw_data_list

        for idx, data_info in enumerate(data_list):
            # Генерация sample_id, если его нет
            if 'sample_id' not in data_info:
                point_cloud_path = data_info['point_cloud']['point_cloud_path']
                filename = osp.basename(point_cloud_path)
                sample_id = osp.splitext(filename)[0]
                data_info['sample_id'] = sample_id
            logging.debug(f"Обработано образец {idx}: sample_id={data_info['sample_id']}")
        
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
        if not ann_info or not ann_info['name']:
            logging.warning("Нет аннотаций для этого образца.")
            return None if not self.test_mode else {
                'gt_bboxes_3d': np.zeros((0, 7), dtype=np.float32),
                'gt_labels_3d': np.array([], dtype=np.int64)
            }

        gt_bboxes_3d = []
        gt_labels_3d = []

        names = ann_info['name']
        locations = ann_info['location']
        dimensions = ann_info['dimensions']
        rotations = ann_info['rotation_y']

        for name, loc, dim, rot in zip(names, locations, dimensions, rotations):
            try:
                label = self.metainfo['classes'].index(name)
            except ValueError:
                logging.error(f"Неизвестный класс: {name}. Пропуск.")
                continue

            loc = np.array(loc, dtype=np.float32)
            dim = np.array(dim, dtype=np.float32)
            rot = float(rot)

            l, w, h = dim
            bbox = [loc[0], loc[1], loc[2], l, w, h, rot]
            gt_bboxes_3d.append(bbox)
            gt_labels_3d.append(label)

        if not gt_bboxes_3d:
            logging.warning("Нет валидных аннотаций после обработки.")
            return None if not self.test_mode else {
                'gt_bboxes_3d': np.zeros((0, 7), dtype=np.float32),
                'gt_labels_3d': np.array([], dtype=np.int64)
            }

        ann = {
            'gt_bboxes_3d': np.array(gt_bboxes_3d, dtype=np.float32),
            'gt_labels_3d': np.array(gt_labels_3d, dtype=np.int64)
        }

        logging.info(f"Обработано {len(gt_bboxes_3d)} аннотаций.")
        return ann
