# model settings
model = 'Digital_twin'
seed=42
model_dict = dict(
    core_dict=dict(
        in_channels=1,
        behavior_dim=4,
        patch_size=[2, 2, 2],
        window_size=[8, 7, 7],
        emb_dim=224,  
        num_blocks=4, 
        num_heads=4, 
        mlp_dim=488,  
        t_dropout=0.,  
        drop_path=0.,
        bias=True,
        grad_checkpointing=False,
    ),
    core_type="Swin3D",
    readout_dict=dict(
        bias=True,
        init_mu_range=0.3, 
        init_sigma=0.2,   
        gauss_type='full',
        grid_mean_predictor={
            'type': 'cortex',
            'input_dimensions': 2,
            'hidden_layers': 1, #
            'hidden_features': 32, 
            'final_tanh': True
        },
        share_features=False,
        share_grid=False,
        shared_match_ids=None,
        gamma_grid_dispersion=0.0,
    ), 
    readout_type='gaussian',
    
    use_shifter=True,
    shifter_dict=dict(
        gamma_shifter=0,
        shift_layers=3,
        input_channels_shifter=2,
        hidden_channels_shifter=16, 
    ),
    shifter_type="MLP",
    
    deeplake_ds=False
)
include_include_behavior_as_channels=False
include_pupil_centers_as_channels=False
video_enhance=False

detach_core = False
scale_loss = False
loss_type = 'possion'

# dataset settings
data_root = '/data/Sensorium_Series/data/Video2activity/'
batch_size = 32
frames = 30

# optimizer and learning rate
optimizer = dict(type='AdamW', lr=0.001, betas=(0.9, 0.999), weight_decay=0.05)
optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy='CosineRestart',
    warmup='linear',
    warmup_iters=600,
    warmup_ratio=1.0 / 3,
    periods=[80, 100],
    restart_weights=[1, 1],
    min_lr=[1e-3, 1e-6]
)


# runtime settings
gpus = [0,1,2,3,4,5,6,7]
dist_params = dict(backend='nccl')
data_workers = 2  # data workers per gpu
checkpoint_config = dict(interval=180)  # save checkpoint at every epoch
workflow = [('train', 1), ('val', 1)]
total_epochs = 180
resume_from = None
load_from = None
work_dir = './work_dir/NeuroEconding_hierachical'

# logging settings
log_level = 'INFO'
log_config = dict(
    interval=30, 
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='WandbLogger'),
    ])
