from torch_geometric.graphgym.register import register_config


@register_config('extended_optim')
def extended_optim_cfg(cfg):
    """Extend optimizer config group that is first set by GraphGym in
    torch_geometric.graphgym.config.set_cfg
    """

    # ReduceLROnPlateau: Factor by which the learning rate will be reduced
    cfg.optim.reduce_factor = 0.1

    # ReduceLROnPlateau: #epochs without improvement after which LR gets reduced
    cfg.optim.schedule_patience = 10

    # ReduceLROnPlateau: Lower bound on the learning rate
    cfg.optim.min_lr = 0.0
