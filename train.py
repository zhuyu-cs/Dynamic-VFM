import logging
import os
from argparse import ArgumentParser
from collections import OrderedDict
import torch
from torch.nn.parallel import DataParallel
from models import make_video_model_adaptive, make_video_model_mean
from utils import mouse_video_loader, compute_loss, get_correlations
import random
import numpy as np
from neuralpredictors.training import LongCycler
from mmcv import Config
from mmcv.runner import Runner
import warnings
warnings.filterwarnings("ignore") 

def get_logger(log_level):
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s', level=log_level)
    logger = logging.getLogger()
    return logger

def parse_args():
    parser = ArgumentParser(description='Train CIFAR-10 classification')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='job launcher')
    return parser.parse_args()

def set_random_seed(seed, deterministic=True):
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


def main():
    args = parse_args()
    for mode in ['0','2_4_8_16', '2_4_8','2_4_16','2_8_16','4_8_16','2_4','2_8','2_16','4_8','4_16','8_16','2', '4','8','16']:
        for readout_type in ['mean', 'adaptive']:
            cfg = Config.fromfile(args.config)

            set_random_seed(cfg.seed)

            logger = get_logger(cfg.log_level)
            
            # build datasets and dataloaders
            num_workers = cfg.data_workers * len(cfg.gpus)
            batch_size = cfg.batch_size
                
            
            mice = ['dynamic29156-11-10-Video-8744edeac3b4d1ce16b680916b5267ce',
                'dynamic29228-2-10-Video-8744edeac3b4d1ce16b680916b5267ce',
                'dynamic29234-6-9-Video-8744edeac3b4d1ce16b680916b5267ce',
                'dynamic29513-3-5-Video-8744edeac3b4d1ce16b680916b5267ce',
                'dynamic29514-2-9-Video-8744edeac3b4d1ce16b680916b5267ce',
                'dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20',      
                'dynamic29623-4-9-Video-9b4f6a1a067fe51e15306b9628efea20',  
                'dynamic29647-19-8-Video-9b4f6a1a067fe51e15306b9628efea20', 
                'dynamic29712-5-9-Video-9b4f6a1a067fe51e15306b9628efea20',   
                'dynamic29755-2-8-Video-9b4f6a1a067fe51e15306b9628efea20',
            ] 
            data_path = [os.path.join(cfg.data_root, m) for m in mice]
            all_loaders = mouse_video_loader(paths = data_path, 
                                            frames = cfg.frames,
                                            num_workers = num_workers,
                                            batch_size = batch_size,
                                            include_include_behavior_as_channels=cfg.include_include_behavior_as_channels,
                                            include_pupil_centers_as_channels=cfg.include_pupil_centers_as_channels,
                                            video_enhance=cfg.video_enhance)
            logger.info('Load dataset Done!')
            # build model 
            if readout_type == 'mean':  
                model = make_video_model_mean(dataloaders=all_loaders,
                            core_dict=cfg.model_dict['core_dict'],
                            core_type=cfg.model_dict['core_type'],
                            readout_dict=cfg.model_dict['readout_dict'],
                            readout_type=cfg.model_dict['readout_type'],
                            use_shifter=cfg.model_dict['use_shifter'],
                            shifter_dict=cfg.model_dict['shifter_dict'],
                            shifter_type=cfg.model_dict['shifter_type'])
            else:
                model = make_video_model_adaptive(dataloaders=all_loaders,
                            core_dict=cfg.model_dict['core_dict'],
                            core_type=cfg.model_dict['core_type'],
                            readout_dict=cfg.model_dict['readout_dict'],
                            readout_type=cfg.model_dict['readout_type'],
                            use_shifter=cfg.model_dict['use_shifter'],
                            shifter_dict=cfg.model_dict['shifter_dict'],
                            shifter_type=cfg.model_dict['shifter_type'])
            logger.info('Build model Done!')
            model = DataParallel(model, device_ids=cfg.gpus).cuda()
            
            def batch_processor(model, data, train_mode):
                data_key, real_data = data
                videos=real_data['videos']
                responses=real_data['responses']
                behavior=real_data['behavior'] 
                pupil_center=real_data['pupil_center']
                videos = videos.cuda(non_blocking=True)
                responses = responses.cuda(non_blocking=True)
                behavior = behavior.cuda(non_blocking=True)
                pupil_center = pupil_center.cuda(non_blocking=True)
                
                model_output = model(inputs=videos,
                                data_key=data_key,
                                behavior=behavior,
                                pupil_center=pupil_center,
                                infer_mode=mode,
                                detach_core=False)
                
                if model.training == True:
                    loss = compute_loss(
                        loss_type=cfg.loss_type,
                        model=model,
                        model_output=model_output,
                        responses=responses,
                        dataloader=all_loaders['train'],
                        data_key=data_key,
                        detach_core=cfg.detach_core,
                        scale_loss=cfg.scale_loss
                    )
                else:
                    loss = compute_loss(
                        loss_type=cfg.loss_type,
                        model=model,
                        model_output=model_output,
                        responses=responses,
                        dataloader=all_loaders['oracle'],
                        data_key=data_key,
                        detach_core=cfg.detach_core,
                        scale_loss=cfg.scale_loss
                    )
                total_loss = loss 
                log_vars = OrderedDict()
                log_vars['whole_loss'] = total_loss.item()
                log_vars['scaled_reconstruct_loss'] = loss.item()
                if model.training == False:
                    skip = responses.shape[2] - model_output.shape[1]
                    validation_correlation = get_correlations(
                                                model_output=model_output,
                                                responses=responses,
                                                as_dict=False,
                                                per_neuron=False,
                                                data_key=data_key,
                                                skip = skip,
                                            )
                    log_vars['correlation'] = validation_correlation.item()
                outputs = dict(loss=loss, log_vars=log_vars, num_samples=videos.shape[0])
                return outputs

            # build runner and register hooks
            runner = Runner(
                model,
                batch_processor,
                cfg.optimizer,
                cfg.work_dir+f'_{readout_type}_{mode}',
                log_level=cfg.log_level)
            runner.register_training_hooks(
                lr_config=cfg.lr_config,
                optimizer_config=cfg.optimizer_config,
                checkpoint_config=cfg.checkpoint_config,
                log_config=cfg.log_config)
            
            if cfg.get('resume_from') is not None:
                runner.resume(cfg.resume_from)
            elif cfg.get('load_from') is not None:
                runner.load_checkpoint(cfg.load_from)

            runner.run([LongCycler(all_loaders['train']), LongCycler(all_loaders['oracle'])], cfg.workflow, cfg.total_epochs)


if __name__ == '__main__':
    main()
