# Setup Guide

## Cloning the repository

To obtain the unifi3d code, use the following command:
```
git clone XXXX
```

## Setting up the virtual environment manager

Setting up a virtual environment manager is highly recommended for an efficient and organized workspace. This guide provides instructions for setting up a virtual environment using either of the following two popular frameworks: [Miniforge](https://github.com/conda-forge/miniforge) and -- alternatively -- [pyenv](https://github.com/pyenv/pyenv) with [venv](https://docs.python.org/3/library/venv.html). 

IMPORTANT: The following process was tested on Ubuntu 20.04, Ubuntu 22.04 and Ubuntu 24.04.

#### Option 1 (recommended): Using miniforge as a virtual environment manager

[Miniforge](https://github.com/conda-forge/miniforge) is a lightweight version of Conda that emphasizes simplicity and has a small footprint. 
To install Miniforge on your system, execute the following commands:
```
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```
It is assumed that miniforge (or mambaforge) is installed on your system within `$HOME/miniforge3`.

#### Option 2: Using pyenv and venv as alternative virtual environment managers

[Pyenv](https://github.com/pyenv/pyenv) is an alternative minimalistic Python version manager, often paired with the virtual environment manager [venv](https://docs.python.org/3/library/venv.html). 
Although `venv` is already part of python3, you may have to install `pyenv` on your system using the following command:
```
git clone https://github.com/pyenv/pyenv.git  ~/.pyenv
git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
```
Once installation is complete, configure your shell to initialize `pyenv` by modifying your `.bashrc` appropirately. You may do so by entering the following commands in a terminal:
```
echo '# >>> pyenv initialize >>>' >> ~/.bashrc
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
echo '# <<< pyenv initialize <<<' >> ~/.bashrc
```
To apply these changes, either source the `.bashrc` file or open a new terminal:
```
source ~/.bashrc
```

IMPORTANT: You may need to install the following packages to be able to use pyenv:
```
sudo apt-get install curl libbz2-dev libreadline-dev
```
In case you are working on a cluster, please contact your administrator.

## Automated installation script (recommended)

We provide a bash script allowing users to automatically setup unifi3d and to create a suitable virtual environment for the project, connected to a specific and well tested version of python. This is the recommended way. 
By default, the script assumes that the [miniforge](https://github.com/conda-forge/miniforge) virtual environment manager is being used. In case you wish to use the [pyenv](https://github.com/pyenv/pyenv)/[venv](https://docs.python.org/3/library/venv.html) altgernative, please change the `ENV_TYPE` variable from `"mamba"` to `"venv"` within `setup_unifi3d.sh`.
To run this script, open a terminal in the `unifi3d` root directory and execute the following commands:
```
chmod a+x setup_unifi3d.sh
./setup_unifi3d.sh
```
The `setup_unifi3d` script ensures that all necessary dependencies are available on the system, automatically installing any that are missing. In case you wish to regenerate the `unifi3d_venv` virtual environment, simply re-execute the `setup_unifi3d` script as follow:
```
./setup_unifi3d.sh true
```

#### Additional dependency: Blender

For data preprocessing, we need Blender to preprocess the data
  - Download Blender locally
    ```
    wget https://download.blender.org/release/Blender4.1/blender-4.1.1-linux-x64.tar.xz
    tar -xvf blender-4.1.1-linux-x64.tar.xz
    ```
  - Update the activation file
    ```
    nano unifi3d_venv/bin/activate
    ```
    add the following lines
    ```
    export BLENDER_PATH=/your/desired/location/blender
    export PATH=$BLENDER_PATH:$PATH
    export LD_LIBRARY_PATH=$BLENDER_PATH/lib:$LD_LIBRARY_PATH
    ```

#### Running Shap-E

Unfortunately, Shap-E is very dependent on specific Pytorch and Blender versions. If you want to run the Shap-E code, please install Blender 3.6.2:
```
wget https://download.blender.org/release/Blender3.6/blender-3.6.2-linux-x64.tar.xz
tar -xvf blender-3.6.2-linux-x64.tar.xz
```
And change the pytorch version in the [install script](setup_unifi3d.sh) to 2.1.0.


## Manual installation (only in case of errors)

In case you prefer setting the repository manually, you may proceed as follow.

### Using miniforge as a virtual environment manager

- First of all create and activate a miniforge environment named `unifi3d_venv` using the provided `environment.yaml` file:
  ```
  mamba create -n "unifi3d_venv" python=3.12.8
  mamba activate unifi3d_venv
  ```

### Using pyenv and venv as alternative virtual environment managers

- First of all make sure you got the right version of python using `pyenv`. In this project, we use `python 3.12`:
  ```
  # If you did not install the desired version of python, run:
  pyenv install 3.12
  #------------------#
  pyenv local 3.12
  python --version
  ```
- Then create and activate a virtual environment named `unifi3d_venv` using `venv`:
  ```
  python -m venv unifi3d_venv
  source unifi3d_venv/bin/activate
  ```

### Common to miniforge and to pyenv/venv 
- Install the relevant dependencies (please replace <platform> with intel, cuda or apple):
  ```
  pip install --upgrade pip
  pip install -r requirements_<platform>.txt
  ```

- Finally install the necessary extension modules using:
  ```
  pip install -e .
  ```
