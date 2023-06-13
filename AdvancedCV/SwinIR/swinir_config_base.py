# Modified by Wang Jianyi for AI6126 Project2
# srresnet_ffhq_300k_(for_MMEditing's_version>=v1.0)
#lahelr Modified to swinIR base

_base_ = 'mmediting/configs/_base_/default_runtime.py'

experiment_name = 'swinir'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

scale = 4

# DistributedDataParallel
model_wrapper_cfg = dict(type='MMSeparateDistributedDataParallel')

# model settings
model = dict(
    type='BaseEditModel',
    generator=dict(
        type='SwinIRNet',
        upscale=scale,
        in_chans=3,
        img_size=128,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6], # base: 6 6s
        embed_dim=60, # base: 180
        num_heads=[6, 6, 6, 6], # base: 6 6s
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection='1conv'),
    pixel_loss=dict(
        type='L1Loss', 
	     loss_weight=1.0, 
	     reduction='mean'),
	 train_cfg=dict(),
    test_cfg=dict(),
    data_preprocessor=dict(
        type='EditDataPreprocessor', 
        mean=[0., 0., 0.], 
        std=[255., 255., 255.]))

load_from="/content/drive/MyDrive/acv_ass_2/swinir_x4s64w8d4e60_8xb4-lr2e-4-500k_div2k-d6622d03.pth"
# load_from = "/content/drive/MyDrive/acv_ass_2/iter_20000.pth"

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='gt',
        channel_order='rgb'),
    dict(type='CopyValues', src_keys=['gt'], dst_keys=['img']),
    dict(
        type='RandomBlur',
        params=dict(
            kernel_size=[41],
            kernel_list=['iso', 'aniso'],
            kernel_prob=[0.5, 0.5],
            sigma_x=[0.2, 5],
            sigma_y=[0.2, 5],
            rotate_angle=[-3.1416, 3.1416],
        ),
        keys=['img'],
    ),
    dict(
        type='RandomResize',
        params=dict(
            resize_mode_prob=[0, 1, 0],  # up, down, keep
            resize_scale=[0.0625, 1],
            resize_opt=['bilinear', 'area', 'bicubic'],
            resize_prob=[1 / 3., 1 / 3., 1 / 3.]),
        keys=['img'],
    ),
    dict(
        type='RandomNoise',
        params=dict(
            noise_type=['gaussian'],
            noise_prob=[1],
            gaussian_sigma=[0, 25],
            gaussian_gray_noise_prob=0),
        keys=['img'],
    ),
    dict(
        type='RandomJPEGCompression',
        params=dict(quality=[50, 95]),
        keys=['img']),
    dict(
        type='RandomResize',
        params=dict(
            target_size=(512, 512),
            resize_opt=['bilinear', 'area', 'bicubic'],
            resize_prob=[1 / 3., 1 / 3., 1 / 3.]),
        keys=['img'],
    ),
    dict(
        type='RandomResize',
        params=dict(
            target_size=(128, 128), resize_opt=['area'], resize_prob=[1]),
        keys=['img'],
    ),
    dict(
        type='Flip', keys=['img', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='SetValues', dictionary=dict(scale=scale)),#added
    dict(type='PackEditInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='PackEditInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='CopyValues', src_keys=['img'], dst_keys=['gt']),
    dict(
        type='RandomResize',
        params=dict(
            target_size = (512,512),
            resize_opt=['area'],
            resize_prob=[1.]),
        keys=['gt'],
    ),
    dict(type='PackEditInputs')
]

# dataset settings
dataset_type = 'BasicImageDataset'

train_dataloader = dict(
    num_workers=6,
    batch_size=6,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(dataset_type='ffhq', task_name='face_sr'),
        data_root='/content/drive/MyDrive/acv_ass_2/data/train/GT',
        data_prefix=dict(gt='', img=''),
        pipeline=train_pipeline))

val_dataloader = dict(
    num_workers=2,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(dataset_type='ffhq', task_name='face_sr'),
        data_root='/content/drive/MyDrive/acv_ass_2/data/val/',
        data_prefix=dict(img='LQ', gt='GT'),
        filename_tmpl=dict(img='{}', gt='{}'),
        pipeline=val_pipeline))

# For test, since you do not have GT, you just need to resize the LQ image using opencv
# and use it as GT. Then you can run the code. The GT is only used to make sure the code
# can run. Remember to change the data root accordingly.
test_dataloader = dict(
    num_workers=2,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(dataset_type='ffhq', task_name='face_sr'),
        data_root='/content/drive/MyDrive/acv_ass_2/data/test/LQ',
        data_prefix=dict(img="", gt=''),
        pipeline=test_pipeline))

val_evaluator = dict(
    type='EditEvaluator', metrics=[
        dict(type='PSNR'),
        dict(type='SSIM'),
    ])
test_evaluator = val_evaluator

max_iters = 50000
val_interval = 5000
lrsn = 5
train_cfg = dict(
    type='IterBasedTrainLoop', 
    max_iters=max_iters, 
    val_interval=val_interval)
val_cfg = dict(type='EditValLoop')
test_cfg = dict(type='EditTestLoop')

# optimizer
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=2e-4, betas=(0.9, 0.999)))

# learning policy, Shangchen added
param_scheduler = dict(
    type='CosineRestartLR',
    by_epoch=False,
    periods=[int(max_iters/lrsn) for _ in range(lrsn)],
    restart_weights=[1 for _ in range(lrsn)],
    eta_min=1e-7)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5000,
        save_optimizer=True,
        by_epoch=False,
        out_dir=save_dir,
    ),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)


# custom hook
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='ConcatImageVisualizer',
    vis_backends=vis_backends,
    fn_key='gt_path',
    img_keys=['pred_img'],#'gt_img', 'input', 
    bgr2rgb=True)
custom_hooks = [
    dict(type='BasicVisualizationHook', interval=1),
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('generator_ema'),
        interval=1,
        interp_cfg=dict(momentum=0.001),
    )
]