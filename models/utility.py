import copy


def get_io_dims(data_loader):
    items = next(iter(data_loader))
    if hasattr(items, "_asdict"):  # if it's a named tuple
        items = items._asdict()
    if hasattr(items, "items"):  # if dict like
        return {k: v.shape for k, v in items.items()}
    else:
        return (v.shape for v in items)


def get_dims_for_loader_dict(dataloaders):

    return {k: get_io_dims(v) for k, v in dataloaders.items()}


def prepare_grid(grid_mean_predictor, dataloaders):
    grid_mean_predictor = copy.deepcopy(grid_mean_predictor)
    grid_mean_predictor_type = grid_mean_predictor.pop("type")

    if grid_mean_predictor_type == "cortex":
        input_dim = grid_mean_predictor.pop("input_dimensions", 2)
        source_grids = {
            k: v.dataset.neurons.cell_motor_coordinates[:, :input_dim]
            for k, v in dataloaders.items()
        }
    return grid_mean_predictor, grid_mean_predictor_type, source_grids
