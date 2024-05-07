import warnings
from typing import Any, Literal, Optional

import torch
from torch import nn as nn
from torch.nn.modules import Module
from torch.nn.parameter import Parameter

Reduction = Literal["sum", "mean", None]


class ConfigurationError(Exception):
    pass


# ------------------ Base Classes -------------------------


class Readout(Module):
    """
    Base readout class for all individual readouts.
    The MultiReadout will expect its readouts to inherit from this base class.
    """

    features: Parameter
    bias: Parameter

    def initialize(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("initialize is not implemented for ", self.__class__.__name__)

    def regularizer(self, reduction: Reduction = "sum", average: Optional[bool] = None) -> torch.Tensor:
        raise NotImplementedError("regularizer is not implemented for ", self.__class__.__name__)

    def apply_reduction(
        self, x: torch.Tensor, reduction: Reduction = "mean", average: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Applies a reduction on the output of the regularizer.
        Args:
            x: output of the regularizer
            reduction: method of reduction for the regularizer. Currently possible are ['mean', 'sum', None].
            average: Deprecated. Whether to average the output of the regularizer.
                            If not None, it is transformed into the corresponding value of 'reduction' (see method 'resolve_reduction_method').

        Returns: reduced value of the regularizer
        """
        reduction = self.resolve_reduction_method(reduction=reduction, average=average)

        if reduction == "mean":
            return x.mean()
        elif reduction == "sum":
            return x.sum()
        elif reduction is None:
            return x
        else:
            raise ValueError(
                f"Reduction method '{reduction}' is not recognized. Valid values are ['mean', 'sum', None]"
            )

    def resolve_reduction_method(self, reduction: Reduction = "mean", average: Optional[bool] = None) -> Reduction:
        """
        Helper method which transforms the old and deprecated argument 'average' in the regularizer into
        the new argument 'reduction' (if average is not None). This is done in order to agree with the terminology in pytorch).
        """
        if average is not None:
            warnings.warn("Use of 'average' is deprecated. Please consider using `reduction` instead")
            reduction = "mean" if average else "sum"
        return reduction

    def resolve_deprecated_gamma_readout(
        self, feature_reg_weight: Optional[float], gamma_readout: Optional[float], default: float = 1.0
    ) -> float:
        if gamma_readout is not None:
            warnings.warn(
                "Use of 'gamma_readout' is deprecated. Use 'feature_reg_weight' instead. If 'feature_reg_weight' is defined, 'gamma_readout' is ignored"
            )

        if feature_reg_weight is None:
            if gamma_readout is not None:
                feature_reg_weight = gamma_readout
            else:
                feature_reg_weight = default
        return feature_reg_weight

    def initialize_bias(self, mean_activity: Optional[torch.Tensor] = None) -> None:
        """
        Initialize the biases in readout.
        Args:
            mean_activity: Tensor containing the mean activity of neurons.

        Returns:

        """
        if mean_activity is None:
            warnings.warn("Readout is NOT initialized with mean activity but with 0!")
            self.bias.data.fill_(0)
        else:
            self.bias.data = mean_activity

    def __repr__(self) -> str:
        return super().__repr__() + " [{}]\n".format(self.__class__.__name__)  # type: ignore[no-untyped-call,no-any-return]


class ClonedReadout(Module):
    """
    This readout clones another readout while applying a linear transformation on the output. Used for MultiDatasets
    with matched neurons where the x-y positions in the grid stay the same but the predicted responses are rescaled due
    to varying experimental conditions.
    """

    def __init__(self, original_readout: Readout, **kwargs: Any) -> None:
        super().__init__()  # type: ignore[no-untyped-call]

        self._source = original_readout
        self.alpha = Parameter(torch.ones(self._source.features.shape[-1]))  # type: ignore[attr-defined]
        self.beta = Parameter(torch.zeros(self._source.features.shape[-1]))  # type: ignore[attr-defined]

    def forward(self, x: torch.Tensor, **kwarg: Any) -> torch.Tensor:
        x = self._source(x) * self.alpha + self.beta
        return x

    def feature_l1(self, average: bool = True) -> torch.Tensor:
        """Regularization is only applied on the scaled feature weights, not on the bias."""
        if average:
            return (self._source.features * self.alpha).abs().mean()
        else:
            return (self._source.features * self.alpha).abs().sum()

    def initialize(self, **kwargs: Any) -> None:
        self.alpha.data.fill_(1.0)
        self.beta.data.fill_(0.0)



class MultiReadoutBase(torch.nn.ModuleDict):
    """
    Base class for MultiReadouts. It is a dictionary of data keys and readouts to the corresponding datasets.
    If parameter-sharing between the readouts is desired, refer to MultiReadoutSharedParametersBase.

    Args:
        in_shape_dict (dict): dictionary of data_key and the corresponding dataset's shape as an output of the core.
        n_neurons_dict (dict): dictionary of data_key and the corresponding dataset's number of neurons
        base_readout (torch.nn.Module): base readout class. If None, self._base_readout must be set manually in the inheriting class's definition
        mean_activity_dict (dict): dictionary of data_key and the corresponding dataset's mean responses. Used to initialize the readout bias with.
                                   If None, the bias is initialized with 0.
        clone_readout (bool): whether to clone the first data_key's readout to all other readouts, only allowing for a scale and offset.
                              This is a rather simple method to enforce parameter-sharing between readouts. For more sophisticated methods,
                              refer to MultiReadoutSharedParametersBase
        gamma_readout (float): regularization strength
        **kwargs:
    """

    _base_readout = None

    def __init__(
        self, in_shape_dict, n_neurons_dict, base_readout=None, mean_activity_dict=None, clone_readout=False, **kwargs
    ):

        # The `base_readout` can be overridden only if the static property `_base_readout` is not set
        if self._base_readout is None:
            self._base_readout = base_readout

        if self._base_readout is None:
            raise ValueError("Attribute _base_readout must be set")
        super().__init__()

        for i, data_key in enumerate(n_neurons_dict):
            first_data_key = data_key if i == 0 else first_data_key
            mean_activity = mean_activity_dict[data_key] if mean_activity_dict is not None else None

            readout_kwargs = self.prepare_readout_kwargs(i, data_key, first_data_key, **kwargs)

            if i == 0 or clone_readout is False:
                self.add_module(
                    data_key,
                    self._base_readout(
                        in_shape=in_shape_dict[data_key],
                        outdims=n_neurons_dict[data_key],
                        mean_activity=mean_activity,
                        **readout_kwargs
                    ),
                )
                original_readout = data_key
            elif i > 0 and clone_readout is True:
                self.add_module(data_key, ClonedReadout(self[original_readout]))

        self.initialize(mean_activity_dict)

    def prepare_readout_kwargs(self, i, data_key, first_data_key, **kwargs):
        return kwargs

    def forward(self, *args, data_key=None, **kwargs):
        if data_key is None and len(self) == 1:
            data_key = list(self.keys())[0]
        return self[data_key](*args, **kwargs)

    def initialize(self, mean_activity_dict=None):
        for data_key, readout in self.items():
            mean_activity = mean_activity_dict[data_key] if mean_activity_dict is not None else None
            readout.initialize(mean_activity)

    def regularizer(self, data_key=None, reduction="sum", average=None):
        if data_key is None and len(self) == 1:
            data_key = list(self.keys())[0]
        return self[data_key].regularizer(reduction=reduction, average=average)


class MultiReadoutSharedParametersBase(MultiReadoutBase):
    """
    Base class for MultiReadouts that share parameters between readouts.
    For more information on which parameters can be shared, refer for example to the FullGaussian2d readout
    """

    def prepare_readout_kwargs(
        self,
        i,
        data_key,
        first_data_key,
        grid_mean_predictor=None,
        grid_mean_predictor_type=None,
        share_transform=False,
        share_grid=False,
        share_features=False,
        **kwargs
    ):
        readout_kwargs = kwargs.copy()

        if grid_mean_predictor:
            if grid_mean_predictor_type == "cortex":
                readout_kwargs["source_grid"] = readout_kwargs["source_grids"][data_key]
                readout_kwargs["grid_mean_predictor"] = grid_mean_predictor
            else:
                raise KeyError("grid mean predictor {} does not exist".format(grid_mean_predictor_type))
            if share_transform:
                readout_kwargs["shared_transform"] = None if i == 0 else self[first_data_key].mu_transform

        elif share_grid:
            readout_kwargs["shared_grid"] = {
                "match_ids": readout_kwargs["shared_match_ids"][data_key],
                "shared_grid": None if i == 0 else self[first_data_key].shared_grid,
            }

        if share_features:
            readout_kwargs["shared_features"] = {
                "match_ids": readout_kwargs["shared_match_ids"][data_key],
                "shared_features": None if i == 0 else self[first_data_key].shared_features,
            }
        else:
            readout_kwargs["shared_features"] = None
        return readout_kwargs
