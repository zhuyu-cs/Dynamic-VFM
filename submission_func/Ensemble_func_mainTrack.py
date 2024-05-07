import torch
import numpy as np
import os
from models import make_video_model_mean, make_video_model_adaptive
from utils import mouse_video_loader 
from utils.mouse_video_loaders_submit import mouse_video_loader_submit
import pandas as pd
import torch
from torch.nn.parallel import DataParallel
import random
import copy 


def set_random_seed(seed=42, deterministic=True):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main_mainTrack(data_root, gpus):
    set_random_seed()
    
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
        readout_dict=dict(bias=True,
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
            align_corners=True,
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
    )
    model_dict_adaptive=copy.deepcopy(model_dict)
    frames=30
    
    mice = [
        'dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20',  
        'dynamic29623-4-9-Video-9b4f6a1a067fe51e15306b9628efea20',    
        'dynamic29647-19-8-Video-9b4f6a1a067fe51e15306b9628efea20',   
        'dynamic29712-5-9-Video-9b4f6a1a067fe51e15306b9628efea20',     
        'dynamic29755-2-8-Video-9b4f6a1a067fe51e15306b9628efea20', 
        
    ] 
    data_path = [os.path.join(data_root, m) for m in mice]
    dataloaders = mouse_video_loader(paths = data_path, 
                                      frames = frames,
                                      num_workers = 4,
                                      batch_size = 1)
    # build model   
    model_mean = make_video_model_mean(dataloaders=dataloaders,
                            core_dict=model_dict['core_dict'],
                            core_type=model_dict['core_type'],
                            readout_dict=model_dict['readout_dict'],
                            readout_type=model_dict['readout_type'],
                            use_shifter=model_dict['use_shifter'],
                            shifter_dict=model_dict['shifter_dict'],
                            shifter_type=model_dict['shifter_type'])
    model_mean = DataParallel(model_mean, device_ids=gpus).cuda()
    model_mean.eval()

    model_adaptive = make_video_model_adaptive(dataloaders=dataloaders,
                            core_dict=model_dict_adaptive['core_dict'],
                            core_type=model_dict_adaptive['core_type'],
                            readout_dict=model_dict_adaptive['readout_dict'],
                            readout_type=model_dict_adaptive['readout_type'],
                            use_shifter=model_dict_adaptive['use_shifter'],
                            shifter_dict=model_dict_adaptive['shifter_dict'],
                            shifter_type=model_dict_adaptive['shifter_type'])
    del dataloaders
    model_adaptive = DataParallel(model_adaptive, device_ids=gpus).cuda()
    model_adaptive.eval()
    
    #! construct the test dataset without 'response'
    mice = [
        'dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20',  
        'dynamic29623-4-9-Video-9b4f6a1a067fe51e15306b9628efea20',    
        'dynamic29647-19-8-Video-9b4f6a1a067fe51e15306b9628efea20',   
        'dynamic29712-5-9-Video-9b4f6a1a067fe51e15306b9628efea20',     
        'dynamic29755-2-8-Video-9b4f6a1a067fe51e15306b9628efea20', 
    ]
    data_path = [os.path.join(data_root, m) for m in mice]

    dataloader = mouse_video_loader_submit(paths = data_path, 
                                      frames = frames,
                                      num_workers = 4,
                                      batch_size = 1,
                                      offset=-1,
                                      )
        
    #! submission setting.
    path='./submission/'
    tier=['live_test', 'final_test']
    track='main'
    skip=50

    assert track == 'main' or track == 'ood', 'Track should be "main" or "ood"'
    if track == 'ood':
        track = 'bonus'
    mice = list(dataloader[list(dataloader.keys())[0]].keys())
    
    if tier is None:
        tier_list = ['live_test', 'final_test']
    else:
        tier_list = tier
                     
    for tier in tier_list:
        tier = f'{tier}_{track}'
        dataframes_pred = []
        for m in mice:
            test_predictions = []
            trial_indices = []
            ds = dataloader[tier][m].dataset
            neuron_ids = np.asarray(ds.neurons.unit_ids.tolist()).astype(np.uint32)
            tiers = ds.trial_info.tiers
            for idx in range(len(tiers)):
                if tiers[idx] == tier:
                    batch = ds.__getitem__(idx)
                    batch_kwargs = batch._asdict() if not isinstance(batch, dict) else batch
                    for bk, bv in batch_kwargs.items():
                        batch_kwargs[bk] = torch.unsqueeze(bv, 0)
                    
                    length = batch_kwargs['videos'].shape[2] - skip #!first 50 frames is deprecated.
                        
                    trial_indices.append(idx)                        
                    videos = batch_kwargs['videos']
                    pupil_center = batch_kwargs['pupil_center']
                    behavior=batch_kwargs['behavior']
                    with torch.no_grad():
                        stacked_input_video = [videos[:, :, batch_index:batch_index+30, :, :] for batch_index in range(35, 299-24, 15)]  
                        stacked_input_video.append(videos[:, :, -39:-9, :, :])    
                        stacked_input_video.append(videos[:, :, -30:, :, :])                          
                        stacked_input_pupil_center = [pupil_center[:, :, batch_index:batch_index+30] for batch_index in range(35, 299-24, 15)]
                        stacked_input_pupil_center.append(pupil_center[:, :, -39:-9])  
                        stacked_input_pupil_center.append(pupil_center[:, :, -30:])  
                        stacked_input_behavior = [behavior[:, :, batch_index:batch_index+30] for batch_index in range(35, 299-24, 15)]
                        stacked_input_behavior.append(behavior[:, :, -39:-9])  
                        stacked_input_behavior.append(behavior[:, :, -30:])  

                        stacked_input_video = torch.stack(stacked_input_video, dim=0).squeeze(1)
                        stacked_input_pupil_center = torch.stack(stacked_input_pupil_center, dim=0).squeeze(1)
                        stacked_input_behavior = torch.stack(stacked_input_behavior, dim=0).squeeze(1)                      
                        
                        all_outs=[]
                        for mode in ['0','2_4_8_16', '2_4_8','2_4_16','2_8_16','4_8_16','2_4','2_8','2_16','4_8','4_16','8_16','2', '4','8','16']:
                            #! mean fusion model
                            if mode=='0':
                                pass
                            else:
                                weight = torch.load(f'./pretrained_weights_mean/model_{mode}.pth')
                                model_mean.load_state_dict(weight, strict=True)
                            
                                out_mean = (model_mean(inputs=stacked_input_video.cuda(), 
                                            pupil_center=stacked_input_pupil_center.cuda(),
                                            behavior=stacked_input_behavior.cuda(),
                                            data_key=m, infer_mode=mode)
                                            .detach()
                                            .cpu())
                                all_outs.append(out_mean)
                            
                            #! adaptive fusion model
                            weight = torch.load(f'./pretrained_weights_adaptive/model_{mode}.pth')
                            model_adaptive.load_state_dict(weight, strict=False)
                    
                            out_adaptive = (model_adaptive(inputs=stacked_input_video.cuda(), 
                                        pupil_center=stacked_input_pupil_center.cuda(),
                                        behavior=stacked_input_behavior.cuda(),
                                        data_key=m, infer_mode=mode)
                                        .detach()
                                        .cpu())
                            all_outs.append(out_adaptive)
                        all_outs = torch.stack(all_outs, dim=0) 
                        out = all_outs.mean(dim=0)
                        real_out = torch.zeros((249, out.shape[-1]))
                        for batch_index, index in zip(range(out.shape[0]), range(0, 249-9, 15)):
                            real_out[index:index+15,...] = out[batch_index, ...]
                        real_out[-9:,...] = out[-1, -9:,...]
                        
                        assert real_out.shape[0] == length, \
                            f'model prediction is too short ({real_out.shape[0]} vs {length})'
                        pred = real_out.permute(1, 0).numpy().astype(np.float32).tolist()
                        test_predictions.append(pred)
                
            df = pd.DataFrame(
                {
                    "mouse": [m] * len(test_predictions),
                    "trial_indices": trial_indices,
                    "prediction": test_predictions,
                    "neuron_ids": [neuron_ids] * len(test_predictions),
                }
            )
            dataframes_pred.append(df) 

        #save file
        df = pd.concat(dataframes_pred, ignore_index=True)
        tier = 'live_main' if 'live_test' in tier else 'final_main'
        submission_filename = f"predictions_{tier}.parquet.brotli"
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, submission_filename) if path is not None else submission_filename
        df.to_parquet(save_path, compression='brotli', engine='pyarrow', index=False)
        print(f"Submission file saved for tier: {tier}, track {track}. Saved in: {save_path}")    
    
if __name__ == '__main__':
    data_root = '/data/Sensorium_Series/data/Video2activity/'
    gpus = [0,1,2,3,4,5,6,7] # default
    main_mainTrack(data_root, gpus)
