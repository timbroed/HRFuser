_base_ = [
    '../_base_/models/cascade_rcnn_hrfuser_fpn_nus_clr_fusion.py',
    '../_base_/datasets/nuscenes_detection_r640_clr_fusion.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_1x.py'
]
model = dict(
    backbone=dict(
        drop_path_rate=0.4,
        extra=dict(
            ModFusionA=dict(
                num_heads=(2, 4),
                num_channels=(78, 156)),
            LidarStageB=dict(
                num_heads=(2,),
                num_channels=(78,)),
            ModFusionB=dict(
                num_heads=(2, 4, 8),
                num_channels=(78, 156, 312)),
            LidarStageC=dict(
                num_heads=(2,),
                num_channels=(78,)),
            ModFusionC=dict(
                num_heads=(2, 4, 8, 16),
                num_channels=(78, 156, 312, 624)),
            # LidarStageD=dict(
                # num_heads=(2,),
                # num_channels=(78,)),
            # ModFusionD=dict(
            #     num_channels=(18, 36, 72, 144)),
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
data=dict(samples_per_gpu=2, workers_per_gpu= 1)
seed=0
