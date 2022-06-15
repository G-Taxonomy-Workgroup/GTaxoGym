import logging

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.utils.device import auto_select_device

from .act import * # noqa
from .config import * # noqa
from .encoder import * # noqa
from .head import * # noqa
from .layer import * # noqa
from .loader import * # noqa
from .loss import * # noqa
from .network import * # noqa
from .optimizer import * # noqa
from .pooling import * # noqa
from .stage import * # noqa
from .train import * # noqa
from .transform import * # noqa

# Set up custom logging levels
logging.addLevelName(15, "DETAIL")
logging.DETAIL = logging.getLevelName("DETAIL")
logging.detail = lambda msg: logging.log(logging.DETAIL, msg)

if cfg.device == "auto":
    auto_select_device()
    logging.warning(f"PyG GraphGym cfg.device was not auto selected correctly,"
                    f" now setting to {cfg.device!r}")
