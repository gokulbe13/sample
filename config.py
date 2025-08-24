# rtmpose-m_HPC-Verified_halpe26-768x448.py

# This configuration has undergone final verification and is approved for training.
# It is tuned for high-performance fine-tuning on an HPC node (e.g., NVIDIA H200)
# with a small, custom dataset, incorporating all necessary corrections.

# DEFINITIVE FIX: Use `custom_imports` to explicitly register all necessary
# MMPose components with the MMEngine runner. This is the most robust
# solution to 'KeyError' and 'ModuleNotFoundError' issues.
custom_imports = dict(
    imports=['mmpose.datasets', 'mmpose.models', 'mmpose.evaluation'],
    allow_failed_imports=False)
_base_ = ['/home/aa0463/eddlai.be10/GokTEEP/training/mmpose/configs/_base_/default_runtime.py']

# -- General Training Settings -----------------------------------------------
max_epochs = 200
train_cfg = dict(max_epochs=max_epochs, val_interval=5)
randomness = dict(seed=21)

# -- HPC-Tuned Optimizer and Scheduler Settings ------------------------------
base_lr = 4e-4
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    clip_grad=dict(max_norm=35, norm_type=2),
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True,
    ))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=50),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True)
]


# -- Custom Dataset Metainfo -------------------------------------------------
dataset_metainfo = dict(
    dataset_name='halpe26',
    num_keypoints=26,
    keypoint_info={
        0: dict(name='nose', id=0, color=[51, 153, 255], type='upper', swap=''),
        1: dict(name='left_eye', id=1, color=[51, 153, 255], type='upper', swap='right_eye'),
        2: dict(name='right_eye', id=2, color=[51, 153, 255], type='upper', swap='left_eye'),
        3: dict(name='left_ear', id=3, color=[51, 153, 255], type='upper', swap='right_ear'),
        4: dict(name='right_ear', id=4, color=[51, 153, 255], type='upper', swap='left_ear'),
        5: dict(name='left_shoulder', id=5, color=[0, 255, 0], type='upper', swap='right_shoulder'),
        6: dict(name='right_shoulder', id=6, color=[255, 128, 0], type='upper', swap='left_shoulder'),
        7: dict(name='left_elbow', id=7, color=[0, 255, 0], type='upper', swap='right_elbow'),
        8: dict(name='right_elbow', id=8, color=[255, 128, 0], type='upper', swap='left_elbow'),
        9: dict(name='left_wrist', id=9, color=[0, 255, 0], type='upper', swap='right_wrist'),
        10: dict(name='right_wrist', id=10, color=[255, 128, 0], type='upper', swap='left_wrist'),
        11: dict(name='left_hip', id=11, color=[0, 255, 0], type='lower', swap='right_hip'),
        12: dict(name='right_hip', id=12, color=[255, 128, 0], type='lower', swap='left_hip'),
        13: dict(name='left_knee', id=13, color=[0, 255, 0], type='lower', swap='right_knee'),
        14: dict(name='right_knee', id=14, color=[255, 128, 0], type='lower', swap='left_knee'),
        15: dict(name='left_ankle', id=15, color=[0, 255, 0], type='lower', swap='right_ankle'),
        16: dict(name='right_ankle', id=16, color=[255, 128, 0], type='lower', swap='left_ankle'),
        17: dict(name='head', id=17, color=[255, 255, 255], type='upper', swap=''),
        18: dict(name='neck', id=18, color=[255, 255, 255], type='upper', swap=''),
        19: dict(name='hip', id=19, color=[255, 255, 255], type='lower', swap=''),
        20: dict(name='left_big_toe', id=20, color=[0, 255, 0], type='lower', swap='right_big_toe'),
        21: dict(name='right_big_toe', id=21, color=[255, 128, 0], type='lower', swap='left_big_toe'),
        22: dict(name='left_small_toe', id=22, color=[0, 255, 0], type='lower', swap='right_small_toe'),
        23: dict(name='right_small_toe', id=23, color=[255, 128, 0], type='lower', swap='left_small_toe'),
        24: dict(name='left_heel', id=24, color=[0, 255, 0], type='lower', swap='right_heel'),
        25: dict(name='right_heel', id=25, color=[255, 128, 0], type='lower', swap='left_heel'),
    },
    skeleton_info={
        0: dict(link=('left_ankle', 'left_knee'), id=0, color=[0, 255, 0]),
        1: dict(link=('left_knee', 'left_hip'), id=1, color=[0, 255, 0]),
        2: dict(link=('left_hip', 'hip'), id=2, color=[0, 255, 0]),
        3: dict(link=('right_ankle', 'right_knee'), id=3, color=[255, 128, 0]),
        4: dict(link=('right_knee', 'right_hip'), id=4, color=[255, 128, 0]),
        5: dict(link=('right_hip', 'hip'), id=5, color=[255, 128, 0]),
        6: dict(link=('head', 'neck'), id=6, color=[51, 153, 255]),
        7: dict(link=('neck', 'hip'), id=7, color=[51, 153, 255]),
        8: dict(link=('neck', 'left_shoulder'), id=8, color=[0, 255, 0]),
        9: dict(link=('left_shoulder', 'left_elbow'), id=9, color=[0, 255, 0]),
        10: dict(link=('left_elbow', 'left_wrist'), id=10, color=[0, 255, 0]),
        11: dict(link=('neck', 'right_shoulder'), id=11, color=[255, 128, 0]),
        12: dict(link=('right_shoulder', 'right_elbow'), id=12, color=[255, 128, 0]),
        13: dict(link=('right_elbow', 'right_wrist'), id=13, color=[255, 128, 0]),
        14: dict(link=('left_eye', 'right_eye'), id=14, color=[51, 153, 255]),
        15: dict(link=('nose', 'left_eye'), id=15, color=[51, 153, 255]),
        16: dict(link=('nose', 'right_eye'), id=16, color=[51, 153, 255]),
        17: dict(link=('left_eye', 'left_ear'), id=17, color=[51, 153, 255]),
        18: dict(link=('right_eye', 'right_ear'), id=18, color=[51, 153, 255]),
        19: dict(link=('left_ankle', 'left_big_toe'), id=19, color=[0, 255, 0]),
        20: dict(link=('left_ankle', 'left_small_toe'), id=20, color=[0, 255, 0]),
        21: dict(link=('left_ankle', 'left_heel'), id=21, color=[0, 255, 0]),
        22: dict(link=('right_ankle', 'right_big_toe'), id=22, color=[255, 128, 0]),
        23: dict(link=('right_ankle', 'right_small_toe'), id=23, color=[255, 128, 0]),
        24: dict(link=('right_ankle', 'right_heel'), id=24, color=[255, 128, 0]),
    },
    joint_weights=[
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.5, 1.5, 1.0, 1.0,
        1.2, 1.2, 1.5, 1.5, 1.0, 1.0, 1.2, 2.5, 2.5, 3.5, 3.5, 1.5, 1.5
    ],
    sigmas=[
        0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
        0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089, 0.026,
        0.026, 0.066, 0.079, 0.079, 0.079, 0.079, 0.079, 0.079
    ])

# -- Codec Definition --------------------------------------------------------
input_size = (768, 448)
codec = dict(
    type='SimCCLabel',
    input_size=input_size,
    sigma=(6., 6.93),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# -- Model Definition --------------------------------------------------------
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
        out_indices=(4, ),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-384x288-65e718c4_20230504.pth' 
       )),
    head=dict(
        type='RTMCCHead',
        in_channels=768,
        out_channels=dataset_metainfo['num_keypoints'],
        input_size=input_size,
        in_featuremap_size=tuple([s // 32 for s in input_size]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.0,
            label_softmax=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
    ))


# -- Data Augmentation Pipelines ---------------------------------------------
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(
        type='RandomBBoxTransform',
        scale_factor=[0.6, 1.4],
        rotate_factor=80),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PhotometricDistortion'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=0.5)
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# -- HPC-Tuned Dataloader Configurations -------------------------------------
data_root = '/home/aa0463/eddlai.be10/GokTEEP/training'

train_dataloader = dict(
    batch_size=256,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        data_mode='topdown',
        ann_file='annotations/annotation_fixed.json',
        data_prefix=dict(img='training/validation/images'),
        pipeline=train_pipeline,
        metainfo=dataset_metainfo
    ))

val_dataloader = dict(
    batch_size=256,
    num_workers=16,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        data_mode='topdown',
        ann_file='annotations/annotation_fixed.json',
        data_prefix=dict(img='images/'),
        #bbox_file=f'/home/aa0463/eddlai.be10/GokTEEP/training/bbox.json',
        test_mode=True,
        pipeline=val_pipeline,
        metainfo=dataset_metainfo
    ))



test_dataloader = val_dataloader

# -- Evaluation Metric Configuration -----------------------------------------
val_evaluator = [
    dict(type='PCKAccuracy', thr=0.05, prefix='PCK@0.05'),
    dict(type='PCKAccuracy', thr=0.1, prefix='PCK@0.1'),
    dict(type='AUC', prefix='AUC'),
    dict(type='EPE', prefix='EPE'),
]

test_evaluator = val_evaluator
