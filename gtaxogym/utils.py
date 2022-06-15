import logging
import time

from yacs.config import CfgNode


class timeit:
    """Timing wrapper.

    Wrap a function by recording its execution time and logging the start and
    end messages.

    """

    def __init__(self, level, start_msg, end_msg=None):
        """Initialize timer.

        Args:
            level (int): logging level.
            start_msg (str): Message to log at the start of the function.
            end_msg (str): Message to log at the end of the function.

        """
        self.log = lambda msg: logging.log(level, msg)
        self.start_msg = start_msg
        self.end_msg = end_msg if end_msg is not None else start_msg

    def __call__(self, func):
        """Wrap function with timer."""

        def timed_func(*args, **kwargs):
            self.log(f'> {self.start_msg}')
            start_time = time.perf_counter()

            results = func(*args, **kwargs)

            duration = time.perf_counter() - start_time
            timestr = time.strftime('%H:%M:%S', time.gmtime(duration)) \
                      + f'{duration:.2f}'[-3:]
            self.log(f'< {self.end_msg} took {timestr}')

            return results

        return timed_func


def flatten_dict(metrics):
    """Flatten a list of train/val/test metrics into one dict to send to wandb.

    Args:
        metrics: List of Dicts with metrics

    Returns:
        A flat dictionary with names prefixed with "train/" , "val/" , "test/"
    """
    prefixes = ['train', 'val', 'test']
    result = {}
    for i in range(len(metrics)):
        # Take the latest metrics.
        stats = metrics[i][-1]
        if stats is not None:
            result.update({f"{prefixes[i]}/{k}": v for k, v in stats.items()})
    return result


def cfg_to_dict(cfg_node, key_list=[]):
    """Convert a config node to dictionary.

    Yacs doesn't have a default function to convert the cfg object to plain
    python dict. The following function was taken from
    https://github.com/rbgirshick/yacs/issues/19
    """
    _VALID_TYPES = {tuple, list, str, int, float, bool}

    if not isinstance(cfg_node, CfgNode):
        if type(cfg_node) not in _VALID_TYPES:
            logging.warning(f"Key {'.'.join(key_list)} with "
                            f"value {type(cfg_node)} is not "
                            f"a valid type; valid types: {_VALID_TYPES}")
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = cfg_to_dict(v, key_list + [k])
        return cfg_dict


def make_wandb_name(cfg):
    # Format dataset name.
    dataset_name = cfg.dataset.format
    if dataset_name.startswith('OGB'):
        dataset_name = dataset_name[3:]
    if dataset_name.startswith('PyG-'):
        dataset_name = dataset_name[4:]
    if dataset_name in ['GNNBenchmarkDataset', 'TUDataset']:
        # Shorten some verbose dataset naming schemes.
        dataset_name = ""
    if cfg.dataset.name != 'none':
        dataset_name += "-" if dataset_name != "" else ""
        dataset_name += cfg.dataset.name
    # Format perturbation name.
    perturbation_name = cfg.perturbation.type
    # Format model name.
    model_name = cfg.model.type
    if cfg.model.type == 'gnn':
        model_name = cfg.gnn.layer_type
    model_name += f".{cfg.name_tag}" if cfg.name_tag else ""
    # Compose wandb run name.
    name = f"{dataset_name}.{perturbation_name}.{model_name}.r{cfg.run_id}"
    return name
