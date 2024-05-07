import torch
import torch.nn as nn
import torch.nn.functional as F

class VideoFiringRateEncoder(nn.Module):
    def __init__(
        self,
        core,
        readout,
        shifter=None
    ):
        """
        An Encoder that wraps the core, readout and optionally a shifter amd modulator into one model.
        The output is one positive value that can be interpreted as a firing rate, for example for a Poisson distribution.
        Args:
            core (nn.Module): Core model. Refer to neuralpredictors.layers.cores
            readout (nn.ModuleDict): MultiReadout model. Refer to neuralpredictors.layers.readouts
            elu_offset (float): Offset value in the final elu non-linearity. Defaults to 0.
            shifter (optional[nn.ModuleDict]): Shifter network. Refer to neuralpredictors.layers.shifters. Defaults to None.
            nonlinearity (str): Non-linearity type to use. Defaults to 'elu'.
            nonlinearity_config (optional[dict]): Non-linearity configuration. Defaults to None.
            gru_module (nn.Module) : the module, which should be called between core and readouts
            twoD_core (boolean) : specifies if the core is 2 or 3 dimensinal to change the input respectively
        """
        super().__init__()
        self.core = core
        self.readout = readout
        self.shifter = shifter
        

    def forward(
        self,
        inputs,
        data_key=None,
        behavior=None,
        pupil_center=None,
        trial_idx=None,
        shift=None,
        infer_mode='0',
        **kwargs,
    ):
    
        x = self.core(inputs, behavior, pupil_center)
        x = torch.transpose(x, 1, 2)
        batch_size = x.shape[0]
        time_points = x.shape[1]

        if self.shifter:
            if pupil_center is None:
                raise ValueError("pupil_center is not given")
            pupil_center = pupil_center[:, :, -time_points:]
            pupil_center = torch.transpose(pupil_center, 1, 2)
            pupil_center = pupil_center.reshape(((-1,) + pupil_center.size()[2:]))
            shift = self.shifter[data_key](pupil_center, trial_idx)
        
        x = x.reshape(((-1,) + x.size()[2:])) # (b*t) c h w
        x = self.readout(x, data_key=data_key, shift=shift, infer_mode=infer_mode, **kwargs)
        
        x = F.elu(x)+1
        
        x = x.reshape(((batch_size, time_points) + x.size()[1:]))
        return x

