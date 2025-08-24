_base_ = ['mmpose/configs/_base_/default_runtime.py']

# Common settings
num_keypoints = 26
input_size = (288, 384)

# Runtime
max_epochs = 80
stage2_num_epochs = 30
base_lr = 4e-3
train_batch_size = 512
val_batch_size = 64

train_cfg = dict(max_epochs=max_epochs, val_interval=10)
randomness = dict(seed=21)

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    clip_grad=dict(max_norm=35, norm_type=2),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# Learning rate scheduler
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0, end=1000),
    dict(type='CosineAnnealingLR', eta_min=base_lr * 0.05, begin=max_epochs // 2,
         end=max_epochs, T_max=max_epochs // 2, by_epoch=True, convert_to_iter_based=True),
]

# Auto scale LR
auto_scale_lr = dict(base_batch_size=1024)

# Codec
codec = dict(
    type='SimCCLabel',
    input_size=input_size,
    sigma=(6., 6.93),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# Model
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.67,
        widen_factor=0.75,
        out_indices=(4,),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-384x288-65e718c4_20230504.pth'
        )
    ),
    head=dict(
        type='RTMCCHead',
        in_channels=768,
        out_channels=num_keypoints,
        input_size=input_size,
        in_featuremap_size=tuple([s // 32 for s in input_size]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256, s=128, expansion_factor=2,
            dropout_rate=0., drop_path=0., act_fn='SiLU',
            use_rel_bias=False, pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.,
            label_softmax=True),
        decoder=codec
    ),
    test_cfg=dict(flip_test=True)
)

# Dataset
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = ''
backend_args = dict(backend='local')

# Import halpe26 metadata inline from Python file
from copy import deepcopy
metainfo = dict(
    dataset_name='halpe26',
    paper_info=dict(
        author='Li, Yong-Lu and Xu, Liang and Liu, Xinpeng and Huang, Xijie'
        ' and Xu, Yue and Wang, Shiyi and Fang, Hao-Shu'
        ' and Ma, Ze and Chen, Mingyang and Lu, Cewu',
        title='PaStaNet: Toward Human Activity Knowledge Engine',
        container='CVPR',
        year='2020',
        homepage='https://github.com/Fang-Haoshu/Halpe-FullBody/',
    ),
    keypoint_info={
        0:
        dict(name='nose', id=0, color=[51, 153, 255], type='upper', swap=''),
        1:
        dict(
            name='left_eye',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap='right_eye'),
        2:
        dict(
            name='right_eye',
            id=2,
            color=[51, 153, 255],
            type='upper',
            swap='left_eye'),
        3:
        dict(
            name='left_ear',
            id=3,
            color=[51, 153, 255],
            type='upper',
            swap='right_ear'),
        4:
        dict(
            name='right_ear',
            id=4,
            color=[51, 153, 255],
            type='upper',
            swap='left_ear'),
        5:
        dict(
            name='left_shoulder',
            id=5,
            color=[0, 255, 0],
            type='upper',
            swap='right_shoulder'),
        6:
        dict(
            name='right_shoulder',
            id=6,
            color=[255, 128, 0],
            type='upper',
            swap='left_shoulder'),
        7:
        dict(
            name='left_elbow',
            id=7,
            color=[0, 255, 0],
            type='upper',
            swap='right_elbow'),
        8:
        dict(
            name='right_elbow',
            id=8,
            color=[255, 128, 0],
            type='upper',
            swap='left_elbow'),
        9:
        dict(
            name='left_wrist',
            id=9,
            color=[0, 255, 0],
            type='upper',
            swap='right_wrist'),
        10:
        dict(
            name='right_wrist',
            id=10,
            color=[255, 128, 0],
            type='upper',
            swap='left_wrist'),
        11:
        dict(
            name='left_hip',
            id=11,
            color=[0, 255, 0],
            type='lower',
            swap='right_hip'),
        12:
        dict(
            name='right_hip',
            id=12,
            color=[255, 128, 0],
            type='lower',
            swap='left_hip'),
        13:
        dict(
            name='left_knee',
            id=13,
            color=[0, 255, 0],
            type='lower',
            swap='right_knee'),
        14:
        dict(
            name='right_knee',
            id=14,
            color=[255, 128, 0],
            type='lower',
            swap='left_knee'),
        15:
        dict(
            name='left_ankle',
            id=15,
            color=[0, 255, 0],
            type='lower',
            swap='right_ankle'),
        16:
        dict(
            name='right_ankle',
            id=16,
            color=[255, 128, 0],
            type='lower',
            swap='left_ankle'),
        17:
        dict(name='head', id=17, color=[255, 128, 0], type='upper', swap=''),
        18:
        dict(name='neck', id=18, color=[255, 128, 0], type='upper', swap=''),
        19:
        dict(name='hip', id=19, color=[255, 128, 0], type='lower', swap=''),
        20:
        dict(
            name='left_big_toe',
            id=20,
            color=[255, 128, 0],
            type='lower',
            swap='right_big_toe'),
        21:
        dict(
            name='right_big_toe',
            id=21,
            color=[255, 128, 0],
            type='lower',
            swap='left_big_toe'),
        22:
        dict(
            name='left_small_toe',
            id=22,
            color=[255, 128, 0],
            type='lower',
            swap='right_small_toe'),
        23:
        dict(
            name='right_small_toe',
            id=23,
            color=[255, 128, 0],
            type='lower',
            swap='left_small_toe'),
        24:
        dict(
            name='left_heel',
            id=24,
            color=[255, 128, 0],
            type='lower',
            swap='right_heel'),
        25:
        dict(
            name='right_heel',
            id=25,
            color=[255, 128, 0],
            type='lower',
            swap='left_heel')
    },
    skeleton_info = {
    0: dict(link=(15, 13), id=0, color=[0, 255, 0]),
    1: dict(link=(13, 11), id=1, color=[0, 255, 0]),
    2: dict(link=(11, 19), id=2, color=[0, 255, 0]),
    3: dict(link=(16, 14), id=3, color=[255, 128, 0]),
    4: dict(link=(14, 12), id=4, color=[255, 128, 0]),
    5: dict(link=(12, 19), id=5, color=[255, 128, 0]),
    6: dict(link=(17, 18), id=6, color=[51, 153, 255]),
    7: dict(link=(18, 19), id=7, color=[51, 153, 255]),
    8: dict(link=(18, 5), id=8, color=[0, 255, 0]),
    9: dict(link=(5, 7), id=9, color=[0, 255, 0]),
    10: dict(link=(7, 9), id=10, color=[0, 255, 0]),
    11: dict(link=(18, 6), id=11, color=[255, 128, 0]),
    12: dict(link=(6, 8), id=12, color=[255, 128, 0]),
    13: dict(link=(8, 10), id=13, color=[255, 128, 0]),
    14: dict(link=(1, 2), id=14, color=[51, 153, 255]),
    15: dict(link=(0, 1), id=15, color=[51, 153, 255]),
    16: dict(link=(0, 2), id=16, color=[51, 153, 255]),
    17: dict(link=(1, 3), id=17, color=[51, 153, 255]),
    18: dict(link=(2, 4), id=18, color=[51, 153, 255]),
    19: dict(link=(15, 20), id=19, color=[0, 255, 0]),
    20: dict(link=(15, 22), id=20, color=[0, 255, 0]),
    21: dict(link=(15, 24), id=21, color=[0, 255, 0]),
    22: dict(link=(16, 21), id=22, color=[255, 128, 0]),
    23: dict(link=(16, 23), id=23, color=[255, 128, 0]),
    24: dict(link=(16, 25), id=24, color=[255, 128, 0])
  },

    # the joint_weights is modified by MMPose Team
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5,
        1.5
    ] + [1., 1., 1.2] + [1.5] * 6,

    # 'https://github.com/Fang-Haoshu/Halpe-FullBody/blob/master/'
    # 'HalpeCOCOAPI/PythonAPI/halpecocotools/cocoeval.py#L245'
    sigmas=[
        0.026,
        0.025,
        0.025,
        0.035,
        0.035,
        0.079,
        0.079,
        0.072,
        0.072,
        0.062,
        0.062,
        0.107,
        0.107,
        0.087,
        0.087,
        0.089,
        0.089,
        0.026,
        0.026,
        0.066,
        0.079,
        0.079,
        0.079,
        0.079,
        0.079,
        0.079,
    ])


# Pipelines
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform', scale_factor=[0.5, 1.5], rotate_factor=90),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PhotometricDistortion'),
    dict(type='Albumentation', transforms=[
        dict(type='Blur', p=0.1),
        dict(type='MedianBlur', p=0.1),
        dict(type='CoarseDropout', max_holes=1, max_height=0.4, max_width=0.4,
             min_holes=1, min_height=0.2, min_width=0.2, p=1.0)
    ]),
    dict(type='GenerateTarget', encoder=codec, use_dataset_keypoint_weights=True),
    dict(type='PackPoseInputs')
]

# Copy & modify pipeline for stage 2
train_pipeline_stage2 = deepcopy(train_pipeline)
for t in train_pipeline_stage2:
    if t.get('type') == 'Albumentation':
        for tf in t['transforms']:
            if tf['type'] == 'CoarseDropout':
                tf['p'] = 0.5

val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# Datasets
dataset_cfg = dict(
    type=dataset_type,
    data_root=data_root,
    data_mode=data_mode,
    ann_file='annotations/ann_halpe26_coco_wholebody.json',
    data_prefix=dict(img='images/'),
    metainfo=metainfo
)

train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=10,
    pin_memory=True,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(**dataset_cfg, pipeline=train_pipeline, test_mode=False)
)

val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(**dataset_cfg, pipeline=val_pipeline, test_mode=True)
)

test_dataloader = val_dataloader

# Hooks
default_hooks = dict(
    checkpoint=dict(save_best='AUC', rule='greater', max_keep_ckpts=1)
)

custom_hooks = [
    dict(type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0002,
         update_buffers=True, priority=49),
    dict(type='mmdet.PipelineSwitchHook',
         switch_epoch=max_epochs - stage2_num_epochs,
         switch_pipeline=train_pipeline_stage2)
]

# Evaluators
test_evaluator = [dict(type='PCKAccuracy', thr=0.1), dict(type='AUC')]
val_evaluator = test_evaluator



# Work directory
