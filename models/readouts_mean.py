import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from .readout_base import (Readout, MultiReadoutSharedParametersBase)
import numpy as np

class FullGaussian2d(Readout):
    """
    A readout using a spatial transformer layer whose positions are sampled from one Gaussian per neuron. Mean
    and covariance of that Gaussian are learned.

    Args:
        in_shape (list, tuple): shape of the input feature map [channels, width, height]
        outdims (int): number of output units
        bias (bool): adds a bias term
        init_mu_range (float): initialises the the mean with Uniform([-init_range, init_range])
                            [expected: positive value <=1]. Default: 0.1
        init_sigma (float): The standard deviation of the Gaussian with `init_sigma` when `gauss_type` is
            'isotropic' or 'uncorrelated'. When `gauss_type='full'` initialize the square root of the
            covariance matrix with with Uniform([-init_sigma, init_sigma]). Default: 1
        batch_sample (bool): if True, samples a position for each image in the batch separately
                            [default: True as it decreases convergence time and performs just as well]
        align_corners (bool): Keyword agrument to gridsample for bilinear interpolation.
                It changed behavior in PyTorch 1.3. The default of align_corners = True is setting the
                behavior to pre PyTorch 1.3 functionality for comparability.
        gauss_type (str): Which Gaussian to use. Options are 'isotropic', 'uncorrelated', or 'full' (default).
        grid_mean_predictor (dict): Parameters for a predictor of the mean grid locations. Has to have a form like
                        {
                        'hidden_layers':0,
                        'hidden_features':20,
                        'final_tanh': False,
                        }
        shared_features (dict): Used when the feature vectors are shared (within readout between neurons) or between
                this readout and other readouts. Has to be a dictionary of the form
               {
                    'match_ids': (numpy.array),
                    'shared_features': torch.nn.Parameter or None
                }
                The match_ids are used to match things that should be shared within or across scans.
                If `shared_features` is None, this readout will create its own features. If it is set to
                a feature Parameter of another readout, it will replace the features of this readout. It will be
                access in increasing order of the sorted unique match_ids. For instance, if match_ids=[2,0,0,1],
                there should be 3 features in order [0,1,2]. When this readout creates features, it will do so in
                that order.
        shared_grid (dict): Like `shared_features`. Use dictionary like
               {
                    'match_ids': (numpy.array),
                    'shared_grid': torch.nn.Parameter or None
                }
                See documentation of `shared_features` for specification.

        source_grid (numpy.array):
                Source grid for the grid_mean_predictor.
                Needs to be of size neurons x grid_mean_predictor[input_dimensions]

    """

    def __init__(
        self,
        in_shape,
        outdims,
        bias,
        init_mu_range=0.1,
        init_sigma=1,
        batch_sample=True,
        align_corners=True,
        gauss_type="full",
        grid_mean_predictor=None,
        shared_features=None,
        shared_grid=None,
        source_grid=None,
        mean_activity=None,
        feature_reg_weight=None,
        gamma_readout=None,  # depricated, use feature_reg_weight instead
        return_weighted_features=False,
        **kwargs,
    ):

        super().__init__()
        self.feature_reg_weight = self.resolve_deprecated_gamma_readout(feature_reg_weight, gamma_readout, default=1.0)
        self.mean_activity = mean_activity
        # determines whether the Gaussian is isotropic or not
        self.gauss_type = gauss_type

        if init_mu_range > 1.0 or init_mu_range <= 0.0 or init_sigma <= 0.0:
            raise ValueError("either init_mu_range doesn't belong to [0.0, 1.0] or init_sigma_range is non-positive")

        # store statistics about the images and neurons
        self.in_shape = in_shape
        self.outdims = outdims

        # sample a different location per example
        self.batch_sample = batch_sample

        # position grid shape
        self.grid_shape = (1, outdims, 1, 2)

        # the grid can be predicted from another grid
        self._predicted_grid = False
        self._shared_grid = False
        self._original_grid = not self._predicted_grid

        if grid_mean_predictor is None and shared_grid is None:
            self._mu = Parameter(torch.Tensor(*self.grid_shape))  # mean location of gaussian for each neuron
        elif grid_mean_predictor is not None and shared_grid is not None:
            raise "Shared grid_mean_predictor and shared_grid_mean cannot both be set"
        elif grid_mean_predictor is not None:
            self.init_grid_predictor(source_grid=source_grid, **grid_mean_predictor)
        elif shared_grid is not None:
            self.initialize_shared_grid(**(shared_grid or {}))

        if gauss_type == "full":
            self.sigma_shape = (1, outdims, 2, 2)
        elif gauss_type == "uncorrelated":
            self.sigma_shape = (1, outdims, 1, 2)
        elif gauss_type == "isotropic":
            self.sigma_shape = (1, outdims, 1, 1)
        else:
            raise ValueError(f'gauss_type "{gauss_type}" not known')

        self.init_sigma = init_sigma
        self.sigma = Parameter(torch.Tensor(*self.sigma_shape))  # standard deviation for gaussian for each neuron

        self.initialize_features(**(shared_features or {}))

        if bias:
            bias = Parameter(torch.Tensor(outdims))
            self.register_parameter("bias", bias)
        else:
            self.register_parameter("bias", None)

        self.init_mu_range = init_mu_range
        self.align_corners = align_corners
        self.initialize(mean_activity)
        self.return_weighted_features = return_weighted_features

        self.up_refine2 = nn.Sequential( 
                nn.Conv1d(1,32, kernel_size=1,padding=0,bias=True),
                nn.ELU(),
                nn.Conv1d(32,1, kernel_size=1,padding=0,bias=False),
        )
        self.up_refine4 = nn.Sequential( 
                nn.Conv1d(1,64, kernel_size=1,padding=0,bias=True),
                nn.ELU(),
                nn.Conv1d(64,1, kernel_size=1,padding=0,bias=False),
        )
        self.up_refine8 = nn.Sequential( 
                nn.Conv1d(1,128, kernel_size=1,padding=0,bias=True),
                nn.ELU(),
                nn.Conv1d(128,1, kernel_size=1,padding=0,bias=False),
        )
        self.up_refine16 = nn.Sequential( 
                nn.Conv1d(1,256, kernel_size=1,padding=0,bias=True),
                nn.ELU(),
                nn.Conv1d(256,1, kernel_size=1,padding=0,bias=False),
        )

    @property
    def shared_features(self):
        return self._features

    @property
    def shared_grid(self):
        return self._mu

    @property
    def features(self):
        if self._shared_features:
            return self.scales * self._features[..., self.feature_sharing_index]
        else:
            return self._features

    @property
    def grid(self):
        return self.sample_grid(batch_size=1, sample=False)

    @property
    def mu_dispersion(self):
        """
        Returns the standard deviation of the learned positions.
        Is used as a regularizer to push neurons to learn similar positions.

        Returns:
            mu_dispersion(float): average dispersion of the mean 2d-position
        """

        return self._mu.squeeze().std(0).sum()

    def feature_l1(self, reduction="sum", average=None):
        """
        Returns l1 regularization term for features.
        Args:
            average(bool): Deprecated (see reduction) if True, use mean of weights for regularization
            reduction(str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
        """
        if self._original_features:
            return self.apply_reduction(self.features.abs(), reduction=reduction, average=average)
        else:
            return 0

    def regularizer(self, reduction="sum", average=None):
        return self.feature_l1(reduction=reduction, average=average) * self.feature_reg_weight

    @property
    def mu(self):
        if self._predicted_grid:
            return self.mu_transform(self.source_grid.squeeze()).view(*self.grid_shape)
        elif self._shared_grid:
            if self._original_grid:
                return self._mu[:, self.grid_sharing_index, ...]
            else:
                return self.mu_transform(self._mu.squeeze())[self.grid_sharing_index].view(*self.grid_shape)
        else:
            return self._mu

    def sample_grid(self, batch_size, sample=None):
        """
        Returns the grid locations from the core by sampling from a Gaussian distribution
        Args:
            batch_size (int): size of the batch
            sample (bool/None): sample determines whether we draw a sample from Gaussian distribution, N(mu,sigma), defined per neuron
                            or use the mean, mu, of the Gaussian distribution without sampling.
                           if sample is None (default), samples from the N(mu,sigma) during training phase and
                             fixes to the mean, mu, during evaluation phase.
                           if sample is True/False, overrides the model_state (i.e training or eval) and does as instructed
        """
        with torch.no_grad():
            self.mu.clamp_(
                min=-1, max=1
            )  # at eval time, only self.mu is used so it must belong to [-1,1] # sigma/variance i    s always a positive quantity

        grid_shape = (batch_size,) + self.grid_shape[1:]

        sample = self.training if sample is None else sample
        if sample:
            norm = self.mu.new(*grid_shape).normal_()
        else:
            norm = self.mu.new(*grid_shape).zero_()  # for consistency and CUDA capability

        if self.gauss_type != "full":
            return torch.clamp(
                norm * self.sigma + self.mu, min=-1, max=1
            )  # grid locations in feature space sampled randomly around the mean self.mu
        else:
            return torch.clamp(
                torch.einsum("ancd,bnid->bnic", self.sigma, norm) + self.mu,
                min=-1,
                max=1,
            )  # grid locations in feature space sampled randomly around the mean self.mu

    def init_grid_predictor(self, source_grid, hidden_features=20, hidden_layers=0, final_tanh=False):
        self._original_grid = False
        layers = [nn.Linear(source_grid.shape[1], hidden_features if hidden_layers > 0 else 2)]

        for i in range(hidden_layers):
            layers.extend(
                [
                    nn.ELU(),
                    nn.Linear(hidden_features, hidden_features if i < hidden_layers - 1 else 2),
                ]
            )

        if final_tanh:
            layers.append(nn.Tanh())
        self.mu_transform = nn.Sequential(*layers)

        source_grid = source_grid - source_grid.mean(axis=0, keepdims=True)
        source_grid = source_grid / np.abs(source_grid).max()
        self.register_buffer("source_grid", torch.from_numpy(source_grid.astype(np.float32)))
        self._predicted_grid = True

    def initialize(self, mean_activity=None):
        """
        Initializes the mean, and sigma of the Gaussian readout along with the features weights
        """
        if mean_activity is None:
            mean_activity = self.mean_activity
        if not self._predicted_grid or self._original_grid:
            self._mu.data.uniform_(-self.init_mu_range, self.init_mu_range)

        if self.gauss_type != "full":
            self.sigma.data.fill_(self.init_sigma)
        else:
            self.sigma.data.uniform_(-self.init_sigma, self.init_sigma)
        self._features.data.fill_(1 / self.in_shape[0])
        self._features2.data.fill_(1 / self.in_shape[0])
        self._features4.data.fill_(1 / self.in_shape[0])
        self._features8.data.fill_(1 / self.in_shape[0])
        self._features16.data.fill_(1 / self.in_shape[0])

        if self._shared_features:
            self.scales.data.fill_(1.0)
        if self.bias is not None:
            self.initialize_bias(mean_activity=mean_activity)

    def initialize_features(self, match_ids=None, shared_features=None):
        """
        The internal attribute `_original_features` in this function denotes whether this instance of the FullGuassian2d
        learns the original features (True) or if it uses a copy of the features from another instance of FullGaussian2d
        via the `shared_features` (False). If it uses a copy, the feature_l1 regularizer for this copy will return 0
        """
        c, w, h = self.in_shape
        self._original_features = True
        if match_ids is not None:
            assert self.outdims == len(match_ids)

            n_match_ids = len(np.unique(match_ids))
            if shared_features is not None:
                assert shared_features.shape == (
                    1,
                    c,
                    1,
                    n_match_ids,
                ), f"shared features need to have shape (1, {c}, 1, {n_match_ids})"
                self._features = shared_features
                self._original_features = False
            else:
                self._features = Parameter(
                    torch.Tensor(1, c, 1, n_match_ids)
                )  # feature weights for each channel of the core
            self.scales = Parameter(torch.Tensor(1, 1, 1, self.outdims))  # feature weights for each channel of the core
            _, sharing_idx = np.unique(match_ids, return_inverse=True)
            self.register_buffer("feature_sharing_index", torch.from_numpy(sharing_idx))
            self._shared_features = True
        else:
            self._features = Parameter(
                torch.Tensor(1, c, 1, self.outdims)
            )  # feature weights for each channel of the core
            self._features2 = Parameter(
                torch.Tensor(1, c, 1, self.outdims//2)
            )  # feature weights for each channel of the core
            self._features4 = Parameter(
                torch.Tensor(1, c, 1, self.outdims//4)
            )  # feature weights for each channel of the core
            self._features8 = Parameter(
                torch.Tensor(1, c, 1, self.outdims//8)
            )  # feature weights for each channel of the core
            self._features16 = Parameter(
                torch.Tensor(1, c, 1, self.outdims//16)
            )  # feature weights for each channel of the core
            self._shared_features = False

    def initialize_shared_grid(self, match_ids=None, shared_grid=None):
        c, w, h = self.in_shape

        if match_ids is None:
            raise "match_ids must be set for sharing grid"
        assert self.outdims == len(match_ids), "There must be one match ID per output dimension"

        n_match_ids = len(np.unique(match_ids))
        if shared_grid is not None:
            assert shared_grid.shape == (
                1,
                n_match_ids,
                1,
                2,
            ), f"shared grid needs to have shape (1, {n_match_ids}, 1, 2)"
            self._mu = shared_grid
            self._original_grid = False
            self.mu_transform = nn.Linear(2, 2)
            self.mu_transform.bias.data.fill_(0.0)
            self.mu_transform.weight.data = torch.eye(2)
        else:
            self._mu = Parameter(torch.Tensor(1, n_match_ids, 1, 2))  # feature weights for each channel of the core
        _, sharing_idx = np.unique(match_ids, return_inverse=True)
        self.register_buffer("grid_sharing_index", torch.from_numpy(sharing_idx))
        self._shared_grid = True

    def forward(self, x, sample=None, shift=None, out_idx=None, infer_mode='0', **kwargs):
        """
        Propagates the input forwards through the readout
        Args:
            x: input data
            sample (bool/None): sample determines whether we draw a sample from Gaussian distribution, N(mu,sigma), defined per neuron
                            or use the mean, mu, of the Gaussian distribution without sampling.
                           if sample is None (default), samples from the N(mu,sigma) during training phase and
                             fixes to the mean, mu, during evaluation phase.
                           if sample is True/False, overrides the model_state (i.e training or eval) and does as instructed
            shift (bool): shifts the location of the grid (from eye-tracking data)
            out_idx (bool): index of neurons to be predicted

        Returns:
            y: neuronal activity
        """
        N, c, w, h = x.size()
        c_in, w_in, h_in = self.in_shape
        if (c_in, w_in, h_in) != (c, w, h):
            raise "the specified feature map dimension is not the readout's expected input dimension"
        feat = self._features.view(1, c, self.outdims)
        feat2 = self._features2.view(1, c, self.outdims//2)
        feat4 = self._features4.view(1, c, self.outdims//4)
        feat8 = self._features8.view(1, c, self.outdims//8)
        feat16 = self._features16.view(1, c, self.outdims//16)

        bias = self.bias
        outdims = self.outdims

        if self.batch_sample:
            # sample the grid_locations separately per image per batch
            grid = self.sample_grid(batch_size=N, sample=sample)  # sample determines sampling from Gaussian
        else:
            # use one sampled grid_locations for all images in the batch
            grid = self.sample_grid(batch_size=1, sample=sample).expand(N, outdims, 1, 2)

        if out_idx is not None:
            if isinstance(out_idx, np.ndarray):
                if out_idx.dtype == bool:
                    out_idx = np.where(out_idx)[0]
            feat = feat[:, :, out_idx]
            grid = grid[:, out_idx]
            if bias is not None:
                bias = bias[out_idx]
            outdims = len(out_idx)
        
        
        if shift is not None:
            grid = grid + shift[:, None, None, :]
            mode='nearest'
            grid2 = F.interpolate(grid.permute(0,2,1,3), size=(self.outdims//2, 2), mode=mode).permute(0,2,1,3)
            grid4 = F.interpolate(grid.permute(0,2,1,3), size=(self.outdims//4, 2), mode=mode).permute(0,2,1,3)
            grid8 = F.interpolate(grid.permute(0,2,1,3), size=(self.outdims//8, 2), mode=mode).permute(0,2,1,3)
            grid16 = F.interpolate(grid.permute(0,2,1,3), size=(self.outdims//16, 2), mode=mode).permute(0,2,1,3)


        y = F.grid_sample(x, grid, align_corners=self.align_corners)
        y2 = F.grid_sample(x, grid2, align_corners=self.align_corners)
        y4 = F.grid_sample(x, grid4, align_corners=self.align_corners)  
        y8 = F.grid_sample(x, grid8, align_corners=self.align_corners)
        y16 = F.grid_sample(x, grid16, align_corners=self.align_corners)
        
        y = (y.squeeze(-1) * feat).sum(1).view(N, outdims)            # B*T N
        y2 = (y2.squeeze(-1) * feat2).sum(1).view(N, outdims//2)      # B*T N
        y4 = (y4.squeeze(-1) * feat4).sum(1).view(N, outdims//4)      # B*T N
        y8 = (y8.squeeze(-1) * feat8).sum(1).view(N, outdims//8)      # B*T N
        y16 = (y16.squeeze(-1) * feat16).sum(1).view(N, outdims//16)  # B*T N
        
        y2_up = F.interpolate(y2.unsqueeze(1).unsqueeze(1), size=(1,outdims), mode='nearest').squeeze(1) #! nearest
        y4_up = F.interpolate(y4.unsqueeze(1).unsqueeze(1), size=(1,outdims), mode='nearest').squeeze(1) #! nearest
        y8_up = F.interpolate(y8.unsqueeze(1).unsqueeze(1), size=(1,outdims), mode='nearest').squeeze(1) #! nearest
        y16_up = F.interpolate(y16.unsqueeze(1).unsqueeze(1), size=(1,outdims), mode='nearest').squeeze(1) #! nearest
        
        y2_up = self.up_refine2(y2_up).squeeze(1)
        y4_up = self.up_refine4(y4_up).squeeze(1)
        y8_up = self.up_refine8(y8_up).squeeze(1)
        y16_up = self.up_refine16(y16_up).squeeze(1)

        
        if infer_mode =='0':
            pass
        elif infer_mode =='2_4_8_16':
            y = (y + y2_up + y4_up + y8_up + y16_up)/5.  
        elif infer_mode =='2_4_8':
            y = (y + y2_up + y4_up + y8_up)/4.
        elif infer_mode =='2_4_16':
            y = (y + y2_up + y4_up + y16_up)/4.
        elif infer_mode =='2_8_16':
            y = (y + y2_up + y8_up + y16_up)/4.
        elif infer_mode =='4_8_16':
            y = (y + y4_up + y8_up + y16_up)/4.
        elif infer_mode =='2_4':
            y = (y + y2_up + y4_up)/3.
        elif infer_mode =='2_8':
            y = (y + y2_up + y8_up)/3.
        elif infer_mode =='2_16':
            y = (y + y2_up + y16_up)/3.
        elif infer_mode =='4_8':
            y = (y + y4_up + y8_up)/3.
        elif infer_mode =='4_16':
            y = (y + y4_up + y16_up)/3.
        elif infer_mode =='8_16':
            y = (y + y8_up + y16_up)/3.        
        elif infer_mode =='2':
            y = (y + y2_up)/2.
        elif infer_mode =='4':
            y = (y + y4_up)/2.
        elif infer_mode =='8':
            y = (y + y8_up)/2.
        elif infer_mode =='16':        
            y = (y + y16_up)/2.
        
        if self.bias is not None:
            y = y + bias
        
        return y


class MultipleFullGaussian2d(MultiReadoutSharedParametersBase):
    _base_readout = FullGaussian2d

