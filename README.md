# Unifi3D: A Study on 3D Representations for Generation and Reconstruction in a Common Framework

This is the code repository for Unifi3D, a unified framework for evaluating the reconstruction and generation performance of 3D representations. We compare these representations based on multiple criteria: quality, computational efficiency, and generalization performance. Beyond standard model benchmarking, our experiments aim to derive best practices over all steps involved in the 3D generation pipeline, including preprocessing, mesh reconstruction, compression with autoencoders, and generation. Our findings highlight that reconstruction errors significantly impact overall performance, underscoring the need to evaluate generation and reconstruction jointly.


### 1.1: Repository structure

<details>
  <summary> Details (Click to expand) </summary>

The Unifi3D repository has the following structure:
```
Unifi3D                                   # The top-level Unifi3D repository.
|                                         #
├── unifi3d                               # Core module regrouping the main functionalities of Unifi3D
|    |                                    #
|    ├── data                             # Torch dataset & sampler classes
|    |
|    ├── losses                           # Loss functions including method specific functions.
|    |   |
|    │   ├── diffusion                    #
|    |   └── autoencoders                 #
|    |
|    ├── models                           # Relevant neural models and related submodules
|    │   │                                #
|    │   ├── diffusion                    #
|    │   ├── autoencoders                 #
|    |   └── modules                      # Representation, personalized network layers, or model componeents
|    |
|    ├── utils                            # Useful functions
|    │   │                                #
|    │   ├── data                         #
|    │   ├── evaluate                     #
|    │   ├── logger                       #
|    │   ├── rendering                    #
|    │   ├── model                        #
|    │   ├── triplane_utils               #
|    |   └── scheduler                    #
|
├── configs                               # Hydra configs
|    |                                    #
|    ├── data                             # Data configs
|    ├── debug                            # Debugging configs
|    ├── experiment                       # Experiment configs
|    ├── extras                           # Extra utilities configs
|    ├── hparams_search                   # Hyperparameter search configs
|    ├── hydra                            # Hydra configs
|    ├── local                            # Local configs
|    ├── logger                           # Logger configs
|    ├── model                            # Model configs
|    ├── paths                            # Project paths configs
|    ├── trainer                          # Trainer configs
|    │                                    #
|    ├── default.yaml                     # Config template
|    ├── eval.yaml                        # Main config for evaluation
|    ├── benchmark.yaml                   # Main config for benchmarking reconstruction
|    ├── sample_diffusion.yaml            # Main config for generation (inference)
|    └── train.yaml                       # Main config for training
|                                         #
├── checkpoints                           # Symbolic links to saved checkpoints in the format of *.safetensors. 
|                                         #
├── data                                  # Torch dataset & sampler classes
│   ├── split_shapenet.csv                # our train-test split of the ShapeNet dataset
│   ├── ShapeNetCore.v1                   # original dataset (NOTE: to be downloaded)
│   └── shapenet_preprocessed             # preprocessed dataset (NOTE: to be created)
|
├── docs                                  # Additional documentation markdown files
|                                         #
├── logs                                  # Result of the training and evaluation runs
|                                         #
├── media                                 # Pictures and other relevant media resources of this repository
|                                         #
├── scripts                               # Python scripts and Jupyter notebooks
|    |                                    #
|    ├── accelerate                       # Main training scripts
|    ├── data                             # Dataset preprocessing scripts
|    ├── evaluate                         # Scripts for evaluation and creating tables and plots
|    ├── diffusion                        # Diffusion inference script
|    └── utils                            # Useful general scripts
│                                         #
├── tests                                 # Unit tests
│                                         #
├── .project-root                         # File for inferring the position of project root directory
├── .python-version                       # Contains the exact version of python used in the project
│                                         #
├── requirements_<platform>.txt           # File containing some of the main dependencies of the project by platform
├── LICENSE                               # License file
├── README.md                             # ReadMe file
└── setup.py                              # File for installing project as a package
```
</details>

### 1.2: Installation & Dependencies

<details>
  <summary> Details (Click to expand) </summary>

We provide a bash script allowing users to automatically setup unifi3d and to create a suitable virtual environment for the project, connected to a specific and well tested version of python. This is the recommended way. Execute the script with:
```
chmod a+x setup_unifi3d.sh
./setup_unifi3d.sh
```

**Note:** Some components, such as dataset creation and user study scripts, require additional dependencies not installed by the setup script:
- Blender 4.1 (ensure it is available in your `PATH`)

For alternative installation methods (using venv, manual step-by-step setup, etc.), refer to our [setup guide](docs/setup_guide.md).

</details>

## 2: Dataset

<details>
  <summary> Details (Click to expand) </summary>

We use the ShapeNet dataset for our experiments. To enable different 3D generation approaches, we preprocess the data into point cloud, SDF etc. 
[Details on generated datasets](docs/generated_datasets.md).

To preprocess the data, follow these steps:
* Download the ShapeNetCore.v1 dataset [here](https://www.shapenet.org/account/). 
* Once downloaded, place it in the `data` directory (there should be a folder `data/ShapeNetCore.v1`).
* Install blender (if not already done) and add it to the `PATH` (see [setup guide](docs/setup_guide.md)).
* Run `python scripts/data/create_preprocessed_dataset.py --task_id 0 --num_tasks 1 data/shapenet_preprocessed` (Note: This can take a while. We recommend to distribute the work to multiple parallel jobs using a large number of CPU cores or multiple machines. Use the `--task_id` and `--num_tasks` to split the work for your parallel compute environment.)

This will save the preprocessed data in `data/shapenet_preprocessed`.

</details>


## 3: Training pipeline

<details>
  <summary> Details (Click to expand) </summary>

- Start by activating the `unifi3d` virtual environment:
  ```bash
  conda activate unifi3d
  ```
- Setup an experiment parameter `.yaml` file in [configs/experiment](./configs/experiment), for instance `my_great_experiment.yaml`. Within this file, set the parameters of your experiment.

- Run the trainer (used for both AE and diffusion training)
  ```bash
  python scripts/accelerate/acc_train.py experiment=my_great_experiment
  ```
  The [training script](./scripts/accelerate/acc_train.py) should start automatically, instanciate a [Trainer](./unifi3d/trainers/acc_trainer.py) and a [Logger](./unifi3d/utils/logger/logger.py) based on the specifications of your experiment parameter file `my_great_experiment.yaml`.
- In case you are still in debug phase and with to see the crash logs, you may use the `HYDRA_FULL_ERROR=1` before your command::
  ```bash
  HYDRA_FULL_ERROR=1 python scripts/accelerate/acc_train.py experiment=my_great_experiment
  ```
- For multi-GPU training, we rely on the [hugging-face accelerate](https://huggingface.co/docs/accelerate/basic_tutorials/launch) pipeline.
  - You may start setting up your accelerate environment by creating an accelerate configuration file:
    ```bash
    accelerate config
    ```
    This will generate a `default_config.yaml` file in `$HOME/.cache/huggingface/accelerate` which will be read by `accelerate` at the beginning of your training session. Simply follow the setup guide and set the number of GPUs according to your hardware.
  - Then on your multi-GPU mahcine or cluster session, you may run the distributed experiment using the following command:
    ```bash
    accelerate launch scripts/accelerate/acc_train.py experiment=my_great_experiment
    ```
  - Note that you can point to a specific config file to change the parameters of accelerate (e.g. for training on a different number of GPUs) using the `--config_file` flag:
    ```bash
    accelerate launch --config_file path/to/config/my_config_file.yaml scripts/accelerate/acc_train.py experiment=my_great_experiment
    ```
  - You might also want to avoid using the `accelerate` config file for greater flexibility. In this case, you may simply use the `--multi_gpu` flag:
    ```bash
    accelerate launch --multi_gpu scripts/accelerate/acc_train.py experiment=my_great_experiment
    ```
    In this case, Accelerate will make some hyperparameter decisions for you, e.g., if GPUs are available, it will use all of them by default without the mixed precision.
  - For a complete list of parameters you can pass in, run:
    ```bash
    accelerate launch -h
    ```
    Check the [hugging-face documentation](https://huggingface.co/docs/accelerate/basic_tutorials/launch) for more details.

Currently, all config files for AE model training (i.e., using `EncoderDecoderTrainer`) have the `acc_` prefix, e.g. [acc_shape2vecset_ae](./configs/experiment/acc_shape2vecset_ae.yaml), while all diffusion models have the `_diffuse` prefix, e.g. [diffuse_sdf_ae_dit](./configs/experiment/diffuse_sdf_ae_dit.yaml).

Most models were trained on 48 GB GPUs (e.g. single GPU with batch size 8 for SDF, Voxel, Triplane & Shap2vecset. DualOctree can be trained with larger batch sizes, Shap-E requires 80GB).

### 3.1 Training encoder models

Example usage:
```bash
python scripts/accelerate/acc_train.py experiment=acc_triplane_ae
python scripts/accelerate/acc_train.py experiment=acc_shape2vecset_ae
python scripts/accelerate/acc_train.py experiment=acc_sdf_ae
python scripts/accelerate/acc_train.py experiment=acc_dualoctree_vae
python scripts/accelerate/acc_train.py experiment=acc_shap_e_ae
```

### 3.2 Training diffusion models

When training a diffusion model, you need to specify the checkpoint of the pretrained autoencoder. For example, train a DiT to denoise an encoded SDF with
```bash
python scripts/accelerate/acc_train.py experiment=diffuse_sdf_dit net_encode.ckpt_path=/path/to/ckpt/model.safetensors
```
Or for DualOctree with VAE and Unet:
```bash
python scripts/accelerate/acc_train.py experiment=diffuse_doctree_vae_unet net_encode.ckpt_path=/path/to/ckpt/model.safetensors
```

### 3.3 Visualize aim stack logging

- Enable tunneling with your desired terminal and set the destination server to `127.0.0.1:43800`:
    ```
    cd ~/unifi3d/
    aim up
    ```

</details>

## 4: Benchmarking reconstruction and generation

<details>
  <summary> Details (Click to expand) </summary>

Do not forget to activate the `unifi3d` virtual environment before running any script:
```bash
conda activate unifi3d # if using miniforge
# OR
# source unifi3d_venv/bin/activate # if using pyenv/venv
```

### 4.1: Benchmark a trained EncoderDecoder

Example usage (start from repository root):
```bash
python scripts/benchmarking.py benchmark=shape2vecset ae_id=ae  # ae_id is one of ae, vae or vqvae
python scripts/benchmarking.py benchmark=sdf ae_id=ae
python scripts/benchmarking.py benchmark=voxel ae_id=vae
```
This command takes the preprocessed ShapeNet objects, encodes and decodes them with the model, and saves the result in the `outputs` folder. The results include
* `results.csv`: Table with metric results and runtimes per sample
* `mesh_comp.obj`: Obj file with two rows of objects, one row showing the ground truth meshes and the back row showing the reconstructed meshes (only the first few to avoid large files)

There is a checkpoint path specified in these `benchmark` configs, but you can overwrite it to test your own checkpoint, e.g. with
```bash
python scripts/benchmarking.py benchmark=sdf ckpt_path=/path/to/checkpoint
python scripts/benchmarking.py benchmark=voxel ckpt_path=/path/to/checkpoint
```
If you used different arguments for training the model, you need to specify those as well:
```bash
python scripts/benchmarking.py benchmark=shape2vecset ckpt_path=/path/to/checkpoint net_encode.layer_norm_encoding=true
```

**Quick testing mode**

Some metrics take quite long to compute, and sometimes this script can be helpful to quickly check whether a trained model works. For testing, set the following parameters in the benchmark [config](configs/benchmark.yaml):
```yaml
- metrics: testing # will only compute two metrics (and sample less points to make them faster)
plot_mesh: True
limit: 10 # will only test on 10 meshes instead of the whole dataset
```

**Collect all results and save latex tables**

To collect all results from all representations, after running the benchmarking script for each of them, there is a script that load the result files and combines them:
```bash
python scripts/evaluate/make_reconstruction_table.py
```
This will directly print the latex code for the table. The output folders for each representation, and their name to display in the table, are currently hardcoded in the script.

### 4.2: Benchmark a diffusion model

Example usage (start from repo root):
```bash
python scripts/diffusion/benchmark_gen.py sample=sdf ae_id=ae diff_id=dit cat_id=Chair
python scripts/diffusion/benchmark_gen.py sample=voxel ae_id=ae diff_id=dit cat_id=Chair
python scripts/diffusion/benchmark_gen.py sample=shape2vecset ae_id=ae diff_id=dit cat_id=Chair
```

These `sample` configs include paths to the currently best checkpoints. You can use your own AE and diffusion checkpoints by setting the corresponding arguments:
```bash
python scripts/diffusion/benchmark_gen.py sample=sdf_vqvae_dit ckpt_path=/change/to/your/diffusion/model.safetensors net_encode.ckpt_path=/change/to/your/ae/model.safetensors
```
Outputs are saved to the `output_path` specified in the config file. The folder will contain the following outputs:
* `results.csv`: Table with runtimes and metrics per sample
* `generated_samples`: Folder with all the obj files for the generated meshes
* `metrics.json`: Json file with the distributional metrics (Cov, MMD, 1-NNA)

**Computing the metrics for data that was already generated**

If you already have a folder with generated samples, and just want to run the generated metrics on it, there is a script that uses a config file and only computes the distribution metrics:
```bash
python scripts/evaluate/compute_uncond_gen_metrics.py sample=sdf_vqvae_dit
```
Again, the metrics are saved in `output_path/metrics.json`

**Collect all results and save latex tables**

To collect all results from all representations, after running the benchmarking script for each of them, there is a script that load the result files and combines them:
```bash
python scripts/evaluate/make_generation_table.py
```
This will directly print the latex code for the table. The output folders for each representation, and their name to display in the table, are currently hardcoded in the script.

### 4.3 Special plots for the ground truth data

We show in our paper that the metrics for unconditional generation require sufficient data to compute. This experiment computes these metrics on the test set of ShapeNet and compare to the train set as a reference, which should result in perfect scores. 
There are two scripts for computing the unconditional metrics on the train and test split of Shapenet.
```
scripts/evaluate/ground_truth_uncond_metrics_compute.py
scripts/evaluate/ground_truth_uncond_metrics_plot.py
```

You can get a description of the parameters by calling the scripts with `--help`.

</details>

## 5: Hyperparameter optimisation

<details>
  <summary> Details (Click to expand) </summary>

We use [Hydra](https://hydra.cc/) and [Optuna](https://optuna.org) to provide key-in-hand hyperparameter optimization capability.

- As for the training and evaluation pipelines, make sure that the `unifi3d` virtual environment is activated:
  ```bash
  conda activate unifi3d
  ```
- Setup an experiment parameter `.yaml` file in [configs/experiment](./configs/experiment), for instance `acc_voxel_vqvae.yaml`. Within this file, set the parameters of your experiment.

- Setup an optuna sweeper parameter `.yaml` file in [configs/hparams_search](./configs/hparams_search), for instance `acc_voxel_vqvae_optuna.yaml`. Within this file, set the hyperparameters you wish to optimize, for instance the learning rate or the batch size. You may use the provided [template](./configs/hparams_search/default_optuna.yaml) as a starting point.

- For multi-GPU hyperparameter optimisation, we rely on the hydra [ray launcher](https://hydra.cc/docs/plugins/ray_launcher/) pluggin to distribute the jobs -- with the parameters sampled by optuna -- to each GPU. In practice you may simply set the value of the `num_parallel_process` parameter to the number of available GPUs. You may use the provided [parallel template](./configs/hparams_search/default_optuna_parallel.yaml) as a starting point.

- Run the optuna hyperparameter optimizer:
  ```bash
  python scripts/accelerate/acc_train.py hparams_search=acc_voxel_vqvae_optuna_parallel experiment=acc_voxel_vqvae
  ```
  The [training script](./scripts/accelerate/acc_train.py) should start automatically, instanciate a [Trainer](./unifi3d/trainers/acc_trainer.py) following the specifications of the Optuna parameter search algorithm, set in `acc_voxel_vqvae_optuna_parallel.yaml`.
- In case you are still in debug phase and with to see the crash logs, you may use the `HYDRA_FULL_ERROR=1` before your command:
  ```bash
  HYDRA_FULL_ERROR=1 python scripts/accelerate/acc_train.py hparams_search=acc_voxel_vqvae_optuna_parallel experiment=acc_voxel_vqvae
  ```

</details>

## 6: User study

<details>
  <summary> Details (Click to expand) </summary>
There are three steps:

1. Prepare the data

This requires blender 4.1 to be in the `PATH`
```bash
python scripts/user_study/prepare_user_study_data.py output_dir=/path/to/output_dir
```

2. Collect data from users

Run
```bash
python scripts/user_study/user_study.py /path/to/output_dir
```
Share the link

3. Compute scores

Open `URL` this will generate `user_results/csv_data/user_study_pairwise_prefs.csv`. Using the download button is not necessary.

Run
```bash
python scripts/evaluate/compute_elo_scores.py --data_dir user_results/csv_data/user_study_pairwise_prefs.csv --exp_dir XXXX --methods doctree_vae_unet_champion sdf_ae_dit voxel_ae_unet shape2vecset_ae_dit shap_e_ae_dit triplane_ae_unet
```
</details>

## 7: Contributing

<details>
  <summary> Details (Click to expand) </summary>

Please check the [contribution guidelines](docs/contribute.md).

</details>


## 8: Relevant Research used in this work

<details>
  <summary> Details (Click to expand) </summary>

As most research projects, Unifi3D stands on the shoulders of giants. Here are some of them:

- [DualOctreeGNN](https://github.com/microsoft/DualOctreeGNN)
- [O-CNN](https://github.com/microsoft/O-CNN)
- [SDFusion](https://yccyenchicheng.github.io/SDFusion/)
- [Shap-E](https://github.com/openai/shap-e/tree/main)
- [Convolutional Occupancy Networks](https://github.com/autonomousvision/convolutional_occupancy_networks/)

</details>

**Relevant Code Frameworks**

<details>
  <summary> Details (Click to expand) </summary>

Part of the code was inspired by the following repositories. We thank the authors of these projects for their great work.

- [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template)
- [Accelerate](https://github.com/huggingface/accelerate)
- [Diffusers](https://github.com/huggingface/diffusers)
- [Hydra](https://hydra.cc/)
- [Optuna](https://optuna.org)
- [pyenv](https://github.com/pyenv/pyenv)
- [Pytorch](https://pytorch.org/)

</details>

## 9: Citation

Reference to our paper will follow soon.

