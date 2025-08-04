# Logging

## Overview

Our logging pipeline is inspired by an [implementation](https://github.com/JiahuiLei/NAP/tree/main/logger) which has been successfully utilized in several well-known projects such as [NAP](https://github.com/JiahuiLei/NAP), [EFEM](https://github.com/JiahuiLei/EFEM) or [LivingScenes](https://github.com/GradientSpaces/LivingScenes). The framework is designed to be both flexible and extensible, supporting a variety of logging mechanisms tailored to different requirements. Below is an overview of the logging module structure:

```
logger                                    # Top-level logging utilities folder
|                                         #
├── implementations                       # Folder containing different logger implementations
|    |                                    #
|    ├── interfaces                       # Folder for logging interfaces
|    |    |                               #
|    |    ├── base_logint.py              # Abstract base logging interface
|    |    ├── aim_logint.py               # AimStack logging interface
|    |    ├── comet_logint.py             # Comet ML logging interface
|    |    ├── mlflow_logint.py            # MLflow logging interface
|    |    ├── neptune_logint.py           # Neptune logging interface
|    |    ├── tensorboard_logint.py       # TensorBoard logging interface
|    |    └── wandb_logint.py             # Weights & Biases logging interface
|    |                                    #
|    ├── base_logger.py                   # Abstract logger base class
|    ├── histogram_logger.py              # Histogram logging implementation
|    ├── hyperparameter_logger.py         # Hyperparameter logging implementation
|    ├── image_logger.py                  # Image logging implementation
|    ├── mesh_logger.py                   # Mesh logging implementation
|    ├── metric_logger.py                 # Metric logging implementation
|    └── video_logger.py                  # Video logging implementation
|                                         #
└── logger.py                             # Main logger implementation.
```
The [`Trainer`](../../trainers/acc_trainer.py) class, which is invoked by the main [`training script`](../../../scripts/accelerate/acc_train.py) takes as argument a list of [`Logger`](./logger.py) objects.
Each `Logger` is responsible for initializing and managing a set of specialized logging routines, housed in dedicated classes (e.g., [`MetricLogger`](./implementations/metric_logger.py), [`ImageLogger`](./implementations/image_logger.py), etc.). These classes inherit from the base class [`BaseLogger`](./implementations/base_logger.py). Furthermore, each `Logger` instance also comes with a *logging interface*, (e.g., [`AimLogint`](./implementations/interfaces/aim_logint.py), [`TensorBoardLogint`](./implementations/interfaces/tensorboard_logint.py), etc.), synergizing with popular logging frameworks.

## Core Logging Functions

A `Logger` exposes two primary logging functions that are utilized by the `Trainer`:
- `log_iter(data)` is called every N *iterations* during the training or evaluation process. It accepts a data dictionary as input and sequentially invokes the `log_iter` function of every logging routines contained within the `Logger`.
- `log_epoch()` is called every M *epochs* during the training or evaluation process. It does not require any arguments and is instead expected to compute and log statistics based on the data collected from previous `log_iter` calls. It also displays relevant information regarding the status of the training run such as GPU utilization. 

At the end of the training session, the logging process should be gracefully terminated using the `end_log` routine. 

**Terminology:**
- the term "*iteration*" refers to the number of times the `forward` method of a neural network is being called. 
- the term "*epoch*" refers to the number of times new data batches are being queried. 
- there are typically multiple *iterations* per *epoch*.

## Detail of the implemented logging routines
<details>
  <summary> Details (Click to expand) </summary>

- `MetricLogger`: used to log various metrics during the training and evaluation phases. Metrics are logged for each batch and summarized at the end of each epoch. This logger can also capture GPU memory consumption if configured to do so.
- `HistogramLogger`: logs histogram data, which can be useful for visualizing the distribution of activations and gradients in the neural network.
- `HyperparameterLogger`: allows logging hyperparameters.
- `MeshLogger`: allows logging meshes using the Open3D or Trimesh libraries.
- `ImageLogger`: saves image data during training and evaluation. This is particularly useful for logging example outputs from generative models.
- `VideoLogger`: saves video data, capturing sequences of images over time. This can be useful for tasks such as video prediction or action recognition.
</details>

## Available interfaces
<details>
  <summary> Details (Click to expand) </summary>

- `AimLogint` provides an [AimStack](https://aimstack.io/) logging interface. 
- `TensorBoardLogint` provides a [TensorBoard](https://www.tensorflow.org/tensorboard) logging interface.
- `CometLogint` provides a [Comet ML](https://www.comet.com) logging interface. 
- `MLFlowLogint` provides an [ML Flow](https://mlflow.org/) logging interface. 
- `NeptuneLogint` provides a [Neptune](https://neptune.ai/) logging interface. 
- `WandbLogint` provides a [Weights & Biases](https://wandb.ai) logging interface. 
</details>

## Example Usage

Below is an example of how to setup a `Logger` in the training process. The setup is managed through the experiment configuration file, located under [`configs/experiment`](../../../configs/experiment)).

To set up logging:
- 1. Set *which quantities you wish to log*.
- 2. Select the logging interface (e.g. [AimStack](https://aimstack.io/) or [TensorBoard](https://www.tensorflow.org/tensorboard)) and configure it appropriately.

This is a typical experiment configuration file:
```yaml
defaults:
  - ...
  # The logging pipeline you wish to use (here AimStack)
  - override /logger: aim  
  - ...

# How you wish to configure the logging pipeline
logger: 
  aim: # Should be the same as /logger
    config: 
      loggers: ["metric", "image", "hist", "hparams"] # List and type of logging implementations 
      tracked_variables: # The keys should match aim/loggers
        metric: ["reconstruction_loss", "regularization_loss"]
        image: ["render_gt", "render_rec_gt", "render_rec"]
        hist: ["hist_loss"]
```
In this example, two quantities of type `metric`, named *"reconstruction_loss"* and *"regularization_loss"* are logged along with three quantities of type `image` named *"render_gt"*, *"render_rec_gt"* and *"render_rec"* and finally one quantity of type `hist` (i.e. histogram) named *"hist_loss"*. 

**Important:** 
- a list of the currently available logging keys (i.e. `metric`, `image`, `hist`, etc.) along with the corresponding module names and class names can be found [here](implementations/__init__.py).
- For each logging key, the framework creates the appropriate logger: e.g., the key `metric` calls the creation of a [`MetricLogger`](./implementations/metric_logger.py) within the `Logger`. Similarly, `image` and `hist` result in the instanciation of a [`ImageLogger`](./implementations/image_logger.py) and of a [`HistLogger`](./implementations/histogram_logger.py).

That's essentially all there is to it! As long as your data batch contains keys that match the descriptions in tracked_variables, you should be able to log these quantities seamlessly.