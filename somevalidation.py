# ---------- Minimal validation-only config for Halpe26 (no training parts) ----------
_base_ = ['mmpose::_base_/default_runtime.py']

# If you prefer to hardcode the checkpoint path here (optional):
# load_from = '/absolute/path/to/your/checkpoint.pth'

# ----- Model (must match your training setup) -----
num_keypoints = 26
input_size = (288, 384)

codec = dict(
    type='SimCCLabel',
    input_size=input_size,
    sigma=(6., 6.93),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False,
)

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
            checkpoint='https://download.openmmlab.com/mmpose/v1/projects/'
                       'rtmposev1/cspnext-m_udp-aic-coco_210e-256x192-f2f7d6f6_20230130.pth'
        )),
    head=dict(
        type='RTMCCHead',
        in_channels=768,
        out_channels=num_keypoints,
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
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
            beta=10.,
            label_softmax=True),
        decoder=codec),
    # Disable flip-test so we don't need flip_pairs
    test_cfg=dict(flip_test=False),
)

# ----- Your exact Halpe26 metadata -----
dataset_info = dict(
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
        17: dict(name='head', id=17, color=[255, 128, 0], type='upper', swap=''),
        18: dict(name='neck', id=18, color=[255, 128, 0], type='upper', swap=''),
        19: dict(name='hip', id=19, color=[255, 128, 0], type='lower', swap=''),
        20: dict(name='left_big_toe', id=20, color=[255, 128, 0], type='lower', swap='right_big_toe'),
        21: dict(name='right_big_toe', id=21, color=[255, 128, 0], type='lower', swap='left_big_toe'),
        22: dict(name='left_small_toe', id=22, color=[255, 128, 0], type='lower', swap='right_small_toe'),
        23: dict(name='right_small_toe', id=23, color=[255, 128, 0], type='lower', swap='left_small_toe'),
        24: dict(name='left_heel', id=24, color=[255, 128, 0], type='lower', swap='right_heel'),
        25: dict(name='right_heel', id=25, color=[255, 128, 0], type='lower', swap='left_heel')
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
    # the joint_weights is modified by MMPose Team
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5, 1.5
    ] + [1., 1., 1.2] + [1.5] * 6,
    sigmas=[
        0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
        0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
        0.026, 0.026, 0.066, 0.079, 0.079, 0.079, 0.079, 0.079, 0.079,
    ],
)

# ----- Data & pipeline (validation only) -----
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'data/halpe26/'  # <- change to your path

backend_args = dict(backend='local')

val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=input_size),
    dict(type='PackPoseInputs'),
]

val_dataloader = dict(
    batch_size=64,
    num_workers=6,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/halpe_val.json',    # <- change if needed
        data_prefix=dict(img='images/'),          # <- change if needed
        test_mode=True,
        pipeline=val_pipeline,
        metainfo=dataset_info,                    # <- your exact metadata
    )
)

test_dataloader = val_dataloader

# ----- Evaluator -----
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/halpe_val.json',
    use_area=False,
    sigmas=dataset_info['sigmas'],               # ensure OKS uses Halpe26 sigmas
)
test_evaluator = val_evaluator

# Keep best checkpoint saving off in pure validation; default_runtime handles logging.
default_hooks = dict()
