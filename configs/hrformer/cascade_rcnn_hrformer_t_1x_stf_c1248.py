_base_ = [
    '../_base_/models/cascade_rcnn_hrformer_fpn_stf.py',
    '../_base_/datasets/kitti_detection_2d_c1248.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_1x.py'
]
model = dict(
    backbone=dict(
        drop_path_rate=0.,
        extra=dict(
            stage2=dict(
                num_channels=(18, 36)),
            stage3=dict(
                num_modules=3,
                num_channels=(18, 36, 72)),
            stage4=dict(
                num_channels=(18, 36, 72, 144)))),
    neck=dict(
        in_channels=[18, 36, 72, 144])
)
# AdamW optimizer, no weight decay for position embedding & layer norm
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
runner=dict(max_epochs=60)
lr_config=dict(policy='step', step=[40,50])
data=dict(samples_per_gpu= 3, workers_per_gpu= 2)
seed=0
