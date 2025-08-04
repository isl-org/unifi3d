from .aim_logint import AimLogint

# from .comet_logint import CometLogint
# from .mlflow_logint import MLFlowLogint
# from .neptune_logint import NeptuneLogint
from .tensorboard_logint import TensorBoardLogint

# from .wandb_logint import WandbLogint


REGISTED_INTERFACE = {
    "aim": AimLogint,
    #   "comet": CometLogint,
    #   "mlflow": MLFlowLogint,
    #   "neptune": NeptuneLogint,
    "tensorboard": TensorBoardLogint,
    # "wandb": WandbLogint,
}
