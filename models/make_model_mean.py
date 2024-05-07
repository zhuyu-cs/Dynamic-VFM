from operator import itemgetter

import torch
from .shifters import MLPShifter

from .readouts_mean import  MultipleFullGaussian2d
from .utility import get_dims_for_loader_dict, prepare_grid
from .video_encoder import VideoFiringRateEncoder
from .swin_core import SwinCore
from contextlib import contextmanager

@contextmanager
def eval_state(model):
    """
    Context manager, within which the model will be under `eval` mode.
    Upon existing, the model will return to whatever training state it
    was as it entered into the context.

    Args:
        model (PyTorch Module): PyTorch Module whose train/eval state is to be managed.

    Yields:
        PyTorch Module: The model switched to eval state.
    """
    training_status = model.training

    try:
        model.eval()
        yield model
    finally:
        model.train(training_status)

def get_module_output(model, input_shape1, input_shape2, input_shape3, use_cuda=True):
    """
    Return the output shape of the model when fed in an array of `input_shape`.
    Note that a zero array of shape `input_shape` is fed into the model and the
    shape of the output of the model is returned.

    Args:
        model (nn.Module): PyTorch module for which to compute the output shape
        input_shape (tuple): Shape specification for the input array into the model
        use_cuda (bool, optional): If True, model will be evaluated on CUDA if available. Othewrise
            model evaluation will take place on CPU. Defaults to True.

    Returns:
        tuple: output shape of the model

    """
    # infer the original device
    initial_device = next(iter(model.parameters())).device
    device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
    with eval_state(model):
        with torch.no_grad():
            input1 = torch.zeros(1, *input_shape1[1:], device=device)
            input2 = torch.zeros(1, *input_shape2[1:], device=device)
            input3 = torch.zeros(1, *input_shape3[1:], device=device)
            output = model.to(device)(input1, input2, input3)
    model.to(initial_device)
    return output.shape

def make_video_model_mean(
    dataloaders,
    core_dict=dict(
        behavior_dim=4,
        patch_size=[2, 4, 4],
        window_size=[8, 7, 7],
        emb_dim=160,
        num_blocks=4,
        num_heads=4,
        mlp_dim=488,
        t_dropout=0.2544, 
        drop_path=0.0,
        bias=True,
        core_reg_scale=0.5379,
        grad_checkpointing=False,
    ),
    core_type="Swin3D",
    readout_dict=dict(), 
    readout_type='attention',
    use_shifter=True,
    shifter_dict=dict(input_channels=2,
        hidden_channels_shifter=2,
        shift_layers=1,
        gamma_shifter=0,
        bias=True),
    shifter_type="MLP"
):
    """
    Model class of a stacked2dCore (from neuralpredictors) and a pointpooled (spatial transformer) readout
    Args:
        dataloaders: a dictionary of dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
        seed: random seed
        grid_mean_predictor: if not None, needs to be a dictionary of the form
            {
            'type': 'cortex',
            'input_dimensions': 2,
            'hidden_layers':0,
            'hidden_features':20,
            'final_tanh': False,
            }
            In that case the datasets need to have the property `neurons.cell_motor_coordinates`
        share_features: whether to share features between readouts. This requires that the datasets
            have the properties `neurons.multi_match_id` which are used for matching. Every dataset
            has to have all these ids and cannot have any more.
        all other args: See Documentation of Stacked2dCore in neuralpredictors.layers.cores and
            PointPooled2D in neuralpredictors.layers.readouts
    Returns: An initialized model which consists of model.core and model.readout
    """

    if "train" in dataloaders.keys():
        dataloaders = dataloaders["train"]

    # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
    batch = next(iter(list(dataloaders.values())[0]))
    in_name, out_name = (
        list(batch.keys())[:2] if isinstance(batch, dict) else batch._fields[:2]
    )
    session_shape_dict = get_dims_for_loader_dict(dataloaders)
    n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
    input_channels = [v[in_name][1] for v in session_shape_dict.values()]

    core_input_channels = (
        list(input_channels.values())[0]
        if isinstance(input_channels, dict)
        else input_channels[0]
    )


    if core_type == "Swin3D":
        core = SwinCore(**core_dict)
    else:
        raise NotImplementedError(f"core type {core_type} is not implemented")
    
    
    if "3D" in core_type and '3D' not in readout_type:
        subselect = itemgetter(0, 2, 3)
        in_shapes_dict = {
            k: subselect(tuple(get_module_output(core, v["videos"], v["behavior"], v["pupil_center"])[1:]))
            for k, v in session_shape_dict.items()
        }
    else:
        subselect = itemgetter(1, 0, 2, 3)
        in_shapes_dict = {
            k: subselect(tuple(get_module_output(core, v["videos"], v["behavior"], v["pupil_center"])[1:]))
            for k, v in session_shape_dict.items()
        }   
    mean_activity_dict = {
        k: next(iter(dataloaders[k]))['responses'].mean(0).mean(-1)
        for k in dataloaders.keys()
    }
    readout_dict["in_shape_dict"] = in_shapes_dict
    readout_dict["n_neurons_dict"] = n_neurons_dict
    readout_dict["loaders"] = dataloaders

    if readout_type == "gaussian":
        grid_mean_predictor, grid_mean_predictor_type, source_grids = prepare_grid(
            readout_dict["grid_mean_predictor"], dataloaders
        )

        readout_dict["mean_activity_dict"] = mean_activity_dict
        readout_dict["grid_mean_predictor"] = grid_mean_predictor
        readout_dict["grid_mean_predictor_type"] = grid_mean_predictor_type
        readout_dict["source_grids"] = source_grids
        readout = MultipleFullGaussian2d(**readout_dict)
    else:
        raise NotImplementedError(f"readout type {readout_type} is not implemented")
    
    gru_module = None

    shifter = None
    if use_shifter:
        data_keys = [i for i in dataloaders.keys()]
        shifter_dict["data_keys"] = data_keys
        if shifter_type == "MLP":
            shifter = MLPShifter(**shifter_dict)
        else:
            raise NotImplementedError(f"shifter type {shifter_type} is not implemented")
    

    model = VideoFiringRateEncoder(
        core=core,
        readout=readout,
        shifter=shifter
    )

    return model
