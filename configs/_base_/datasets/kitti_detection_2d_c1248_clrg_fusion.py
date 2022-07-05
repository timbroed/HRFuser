dataset_type = 'Kitti2DDataset'
data_root = 'data/dense/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
input_modality = dict(use_lidar=False, use_camera=True)
img_norm_cfg = dict(
    mean=[95.07200648, 91.35659045, 87.7264499], std=[42.78716034, 42.98587388, 43.82545466], to_rgb=True)
gated_norm_cfg = dict(
    mean=[181.74427536], std=[185.49071888], to_rgb=False)
lidar_norm_cfg = dict(
    mean=[0.014311949, 0.39251423, 3.4071422], std=[0.17276553984335935, 3.76054903771461, 26.008978714330535], to_rgb=False) # rih values calculated over full iamge
radar_norm_cfg = dict(
    mean=[3.4423912, 0.021001821], std=[19.330362993097626, 0.7612592077132296], to_rgb=False) # riv values calculated over full iamge
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadProjectedSensorImageFile', expected_shape=(768, 1280, 3) , sensor_type='lidar', to_float32=True, color_type='unchanged', channels=['yzi']), #,'xz0']
    dict(type='Normalize', **lidar_norm_cfg, keys=['lidar_img'], sensor_type='lidar'),
    dict(type='LoadProjectedSensorImageFile', expected_shape=(768, 1280, 3), sensor_type='radar', to_float32=True, color_type='unchanged', channels=['yzv'], delete_channels=[0]), #,'xz0'] y=height, y=depth
    dict(type='Normalize', **radar_norm_cfg, keys=['radar_img'], sensor_type='radar'),
    dict(type='LoadGatedImageFromFile', gated_folders = ['gated_acc_wraped_grey'], to_float32=True, color_type='unchanged'),
    dict(type='Normalize', **gated_norm_cfg, keys=['gated_img'], sensor_type='gated'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Crop', crop_size=(768, 1280), offsets=(202, 280), skip_keys=['lidar_img', 'radar_img', 'gated_img']),
    dict(type='Resize', img_scale=(1280, 768), keep_ratio=False, skip_keys=['lidar_img', 'radar_img', 'gated_img']),
    dict(type='Crop', crop_size=(384, 1248), offsets=(192, 16), thresh_in_frame=0.1),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg, keys=['img']),
    dict(type='Pad', size_divisor=32),
    dict(type='RandomDrop', p = [0.5, 0.5, 0.5, 0.5], keys=['img', 'lidar_img', 'radar_img', 'gated_img']),
    dict(type='DefaultFormatBundle', sensor_keys=['img', 'lidar_img', 'radar_img', 'gated_img']),    
    dict(type='Collect', keys=['img', 'lidar_img', 'radar_img', 'gated_img','gt_bboxes', 'gt_labels'],
    meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'img_norm_cfg', 
                            'lidar_ori_shape', 'lidar_norm_cfg', 
                            'radar_ori_shape', 'radar_norm_cfg',
                            'gated_ori_shape')),
]

test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadProjectedSensorImageFile', expected_shape=(768, 1280, 3) , sensor_type='lidar', to_float32=True, color_type='unchanged', channels=['yzi']), #,'xz0']
    dict(type='LoadProjectedSensorImageFile', expected_shape=(768, 1280, 3), sensor_type='radar', to_float32=True, color_type='unchanged', channels=['yzv'], delete_channels=[0]), #,'xz0'] y=height, y=depth
    dict(type='LoadGatedImageFromFile', gated_folders = ['gated_acc_wraped_grey'], to_float32=True, color_type='unchanged'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 768),
        flip=False,
        transforms=[
            dict(type='Normalize', **lidar_norm_cfg, keys=['lidar_img'], sensor_type='lidar'),
            dict(type='Normalize', **radar_norm_cfg, keys=['radar_img'], sensor_type='radar'),
            dict(type='Normalize', **gated_norm_cfg, keys=['gated_img'], sensor_type='gated'),
            dict(type='Crop', crop_size=(768, 1280), offsets=(202, 280), skip_keys=['lidar_img', 'radar_img', 'gated_img']),
            dict(type='Resize', keep_ratio=False, skip_keys=['lidar_img', 'radar_img', 'gated_img']),
            dict(type='Normalize', **img_norm_cfg, keys=['img'], sensor_type='img'),
            dict(type='Crop', crop_size=(384, 1248), offsets=(192, 16), thresh_in_frame=0.1),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img', 'lidar_img', 'radar_img', 'gated_img']),
            dict(type='Collect', keys=['img', 'lidar_img', 'radar_img', 'gated_img'],
                     meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'img_norm_cfg', 'crop_factor')),
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
        lidar_prefix='',
        radar_prefix='',
        lidar_img_mode=True,
        radar_img_mode=True,
        classes=class_names,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file= 'dense_infos_val_clear.pkl',
        img_prefix='',
        lidar_prefix='',
        radar_prefix='',
        lidar_img_mode=True,
        radar_img_mode=True,
        classes=class_names,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file= ['dense_infos_test_clear.pkl',
        'dense_infos_light_fog.pkl',
        'dense_infos_dense_fog.pkl',
        'dense_infos_snow.pkl'],
        img_prefix='',
        lidar_prefix='',
        radar_prefix='',
        lidar_img_mode=True,
        radar_img_mode=True,
        classes=class_names,
        pipeline=test_pipeline))
evaluation = dict(
    interval=1, 
    eval_on_crop=dict(
        offset_h=394,
        offset_w=296,
        img_shape=(384, 1248),
        thresh_in_frame=0.1))