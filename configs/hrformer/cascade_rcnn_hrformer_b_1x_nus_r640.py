_base_ = [
    '../_base_/models/cascade_rcnn_hrformer_fpn_nus.py',
    '../_base_/datasets/nuscenes_detection_r640.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_1x.py'
]
model = dict(
    backbone=dict(
        drop_path_rate=0.4,
        extra=dict(
            stage2=dict(
                num_heads=(2, 4),
                num_channels=(78, 156)),
            stage3=dict(
                num_heads=(2, 4, 8),
                num_channels=(78, 156, 312)),
            stage4=dict(
                num_heads=(2, 4, 8, 16),
                num_channels=(78, 156, 312, 624)))),
    neck=dict(
        in_channels=[78, 156, 312, 624])
)
# AdamW optimizer, no weight decay for position embedding & layer norm
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0003,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
data=dict(samples_per_gpu= 4, workers_per_gpu= 2)
seed=0
