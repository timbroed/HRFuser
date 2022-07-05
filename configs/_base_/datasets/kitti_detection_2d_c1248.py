dataset_type = 'Kitti2DDataset'
data_root = 'data/dense/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
input_modality = dict(use_lidar=False, use_camera=True)
img_norm_cfg = dict(
    mean=[95.07200648, 91.35659045, 87.7264499], std=[42.78716034, 42.98587388, 43.82545466], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Crop', crop_size=(768, 1280), offsets=(202, 280)),
    dict(type='Resize', img_scale=(1280, 768), keep_ratio=False),
    dict(type='Crop', crop_size=(384, 1248), offsets=(192, 16), thresh_in_frame=0.1),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg, keys=['img']),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 768),
        flip=False,
        transforms=[
            dict(type='Crop', crop_size=(768, 1280), offsets=(202, 280)),
            dict(type='Resize', keep_ratio=False),
            dict(type='Crop', crop_size=(384, 1248), offsets=(192, 16), thresh_in_frame=0.1),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file= 'dense_infos_train_clear.pkl', # For debugging: dense_infos_debug.pkl'
        img_prefix='',
        classes=class_names,
        pipeline=train_pipeline,
        test_mode=False,),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file= 'dense_infos_val_clear.pkl',
        img_prefix='',
        classes=class_names,
        pipeline=test_pipeline,
        test_mode=True,),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file= ['dense_infos_test_clear.pkl',
        'dense_infos_light_fog.pkl',
        'dense_infos_dense_fog.pkl',
        'dense_infos_snow.pkl'],
        img_prefix='',
        classes=class_names,
        pipeline=test_pipeline,
        test_mode=True,))
evaluation = dict(
    interval=1, 
    eval_on_crop=dict(
        offset_h=394,
        offset_w=296,
        img_shape=(384, 1248),
        thresh_in_frame=0.1))