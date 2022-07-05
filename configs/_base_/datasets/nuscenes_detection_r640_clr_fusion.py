# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/nuscenes/'
class_names = [
        'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
        'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
    ]
classes = [
        'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
        'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
    ]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
lidar_norm_cfg = dict(
    mean=[0.23277158, 0.31501067, -0.00012928071], std=[2.5538357826888602, 3.7345728854535643, 0.2815488539921788], to_rgb=False) # rih values calculated over full iamge
radar_norm_cfg = dict(
    mean=[0.19778967, 0.03477772, 0.0025186215], std=[3.219927182957935, 0.7240392925308506, 0.11561270078715341], to_rgb=False) # riv values calculated over full iamge
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadProjectedSensorImageFile', sensor_type='lidar', to_float32=True, color_type='unchanged', channels=['rih']), #,'xz0']
    dict(type='Normalize', **lidar_norm_cfg, keys=['lidar_img'], sensor_type='lidar'),
    dict(type='LoadProjectedSensorImageFile', sensor_type='radar', to_float32=True, color_type='unchanged', channels=['riv']), #,'xz0']
    dict(type='Normalize', **radar_norm_cfg, keys=['radar_img'], sensor_type='radar'), #, with_mask='radar_mask'
    dict(type='LoadAnnotations', with_bbox=True, with_visibility=True),
    dict(type='Resize', img_scale=(640, 360), keep_ratio=True, skip_keys=['lidar_img', 'radar_img']),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg, keys=['img']),
    dict(type='Pad', size_divisor=32),
    dict(type='RandomDrop', p = [0.2, 0.2, 0.2], keys=['img', 'lidar_img', 'radar_img']),
    dict(type='DefaultFormatBundle', sensor_keys=['img', 'lidar_img', 'radar_img']),
    dict(type='Collect', keys=['img', 'lidar_img', 'radar_img','gt_bboxes', 'gt_labels'],
    meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'img_norm_cfg', 'lidar_ori_shape', 'lidar_norm_cfg', 'radar_ori_shape', 'radar_norm_cfg')),

]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadProjectedSensorImageFile', sensor_type='lidar', to_float32=True, color_type='unchanged', channels=['rih']),
    dict(type='LoadProjectedSensorImageFile', sensor_type='radar', to_float32=True, color_type='unchanged', channels=['riv']),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 360),
        flip=False,
        transforms=[
            dict(type='Normalize', **lidar_norm_cfg, keys=['lidar_img'], sensor_type='lidar'),
            dict(type='Normalize', **radar_norm_cfg, keys=['radar_img'], sensor_type='radar'),
            dict(type='Resize', keep_ratio=True, skip_keys=['lidar_img', 'radar_img']),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg, keys=['img']),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img', 'lidar_img', 'radar_img']),
            dict(type='Collect', keys=['img', 'lidar_img', 'radar_img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        data_root=data_root,
        ann_file='nuscenes_infos_train_mono3d.coco.json',
        img_prefix='',
        lidar_prefix='',
        radar_prefix='',
        lidar_img_mode=True,
        radar_img_mode=True,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        data_root=data_root,
        ann_file='nuscenes_infos_val_mono3d.coco.json',
        img_prefix='',
        lidar_prefix='',
        radar_prefix='',
        lidar_img_mode=True,
        radar_img_mode=True,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        # In case we want to evaluate on a subset of classes:
        # evaluation_ids=[0, 1, 3, 5, 6, 7], # 2: trailer  4: construction_vehicle 8: traffic 9: barrier
        data_root=data_root,
        ann_file='nuscenes_infos_val_mono3d.coco.json',
        img_prefix='',
        lidar_prefix='',
        radar_prefix='',
        lidar_img_mode=True,
        radar_img_mode=True,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')