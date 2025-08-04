#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e
IFS=$'\n\t'

# Check if the current shell is bash
if [ -z "$BASH_VERSION" ]; then
    echo "Error: This script must be run in bash." >&2
    exit 1
fi

# Global variables
readonly PYTHON_VERSION="3.12.8"
readonly ENV_NAME=unifi3d

################################################################################
#  Helper functions                                                            #
################################################################################

# Logging function
log() {
    local type=$1
    local message=$2
    local len=${#message}
    local border=$(printf '%*s' "$((len + 4))" | tr ' ' '#')

    # Colors
    local NC="\033[0m" # No Color
    local GREEN="\033[0;32m"
    local YELLOW="\033[0;33m"
    local RED="\033[0;31m"

    # Choosing color based on message type
    local COLOR="$NC"
    case "$type" in
        DEBUG) COLOR="$NC" ;;
        INFO) COLOR="$GREEN" ;;
        WARNING) COLOR="$YELLOW" ;;
        ERROR) COLOR="$RED" ;;
    esac

    # Printing the message
    echo -e "${COLOR}${border}${NC}"
    echo -e "${COLOR}# $message #${NC}"
    echo -e "${COLOR}${border}${NC}"
}

# Detect project root
find_root() {
    local dir=${1:-"$PWD"}
    while [[ $dir != / ]]; do
        [[ -e "$dir/.project-root" ]] && { printf '%s\n' "$dir"; return; }
        dir=$(dirname "$dir")
    done
    return 1
}

# Detect conda prefix
detect_conda_prefix() {
    for d in "$HOME"/{mambaforge,miniforge3,anaconda3}; do
        [[ -d $d ]] && { echo "$d"; return; }
    done
    log ERROR "No valid Miniforge or Anaconda installation found." && exit 1
}

# Activate conda environment
activate_env() {
    local prefix; prefix=$(detect_conda_prefix)
    log INFO  "Activating $ENV_NAME from $prefix"
    # shellcheck disable=SC1090
    source "$prefix/bin/activate" "$ENV_NAME"
}

# Handle SIGINT
handle_sigint() {
    log "WARNING" "SIGINT received, cleaning up..."
    exit 1  # Exit the script
}

# Run tests
run_tests() {
    log "INFO" "Running Unittests..."
    pushd tests
    pytest test_torch_env.py || { log "ERROR" "Tests failed"; exit 1; }
    popd
}

################################################################################
#  Setup conda environment                                                     #
################################################################################
setup_conda_environment() {
    log "INFO" "Setting up Python virtual environment $ENV_NAME with conda"
    echo "Checking conda installation."
    if ! command -v conda &> /dev/null; then
        log "ERROR" "conda could not be found, please install it first."
        exit 1
    fi

    # Check if the $ENV_NAME environment exists
    if conda env list | grep -q "$ENV_NAME"; then
        echo "$ENV_NAMEenvironment already exists."
        local regenerate=${1:-false}
        if [ "$regenerate" = true ]; then
            echo "Regenerating $ENV_NAME environment."
            conda remove -n $ENV_NAME --all -y
            conda create -n $ENV_NAME python=$PYTHON_VERSION -y
        fi
    else
        echo "$ENV_NAME environment doesn't exist."
        echo "Creating the $ENV_NAME environment."
        conda create -n $ENV_NAME python=$PYTHON_VERSION -y
    fi
    echo "Activating $ENV_NAME environment."
    activate_env
}

################################################################################
#  Install the required python packages                                        #                          
################################################################################
install_required_packages() {
    ### UPGRADE PIP ###
    log "INFO" "Upgrading pip"
    pip install --upgrade pip || { log "ERROR" "Failed to upgrade pip"; exit 1; }

    ### DETECT OPERATING SYSTEM ###
    OS_TYPE=$(uname)
    log "INFO" "Detected OS: $OS_TYPE"

    ### DETECT SYSTEM ARCHITECTURE ###
    ARCH_TYPE=$(uname -m)
    log "INFO" "Detected Architecture: $ARCH_TYPE"

    ### SELECT REQUIREMENTS FILE BASED ON OS AND ARCHITECTURE ###
    if [[ "$OS_TYPE" == "Darwin" ]]; then
        if [[ "$ARCH_TYPE" == "arm64" ]]; then
            REQUIREMENTS_FILE="requirements_apple.txt"
            log "INFO" "Selected requirements file for macOS ARM (Apple Silicon)."
        else
            log "ERROR" "Unsupported architecture on macOS: $ARCH_TYPE"
            exit 1
        fi
    elif [[ "$OS_TYPE" == "Linux" ]]; then
        if [[ "$ARCH_TYPE" == "x86_64" ]]; then
            # Further detect if Intel AI silicon is present
            if lscpu | grep -i "Intel" &>/dev/null && command -v intel_gpu_top &>/dev/null; then
                REQUIREMENTS_FILE="requirements_intel.txt"
                # Please check https://dgpu-docs.intel.com/driver/client/overview.html#ubuntu-24.10-24.04
                log "INFO" "Selected requirements file for Linux with Intel AI silicon."
            else
                REQUIREMENTS_FILE="requirements_cuda.txt"
                log "INFO" "Selected requirements file for Linux with CUDA."
            fi
        else
            log "ERROR" "Unsupported architecture on Linux: $ARCH_TYPE"
            exit 1
        fi
    else
        log "ERROR" "Unsupported OS: $OS_TYPE"
        exit 1
    fi

    ### INSTALL REQUIRED PACKAGES ###
    if [ -f "$REQUIREMENTS_FILE" ]; then
        log "INFO" "Installing Python packages from $REQUIREMENTS_FILE"
        pip install --use-pep517 -r "$REQUIREMENTS_FILE" || { log "ERROR" "Failed to install packages from $REQUIREMENTS_FILE"; exit 1; }
    else
        log "ERROR" "$REQUIREMENTS_FILE not found."
        exit 1
    fi

    ### Triplane utils
    cd unifi3d/utils/triplane_utils
    python setup.py build_ext --inplace
    cd ../../..
}

################################################################################
#  Build the package                                                           #   
################################################################################
build_package() {
    log "INFO" "Checking compiler"
    local CC
    local CXX
    CC=$(which gcc)
    CXX=$(which g++)

    # Check GCC version
    $CC --version || { log "ERROR" "Failed to find a suitable C compiler"; exit 1; }

    # Check G++ version
    $CXX --version || { log "ERROR" "Failed to find a suitable C++ compiler"; exit 1; }

    # Build the Manifold tool
    #log "INFO" "Building the manifold packages"
    #if [ -d "unifi3d/utils/Manifold/build" ]; then
    #    rm -r unifi3d/utils/Manifold/build
    #fi
    #mkdir -p unifi3d/utils/Manifold/build
    #pushd unifi3d/utils/Manifold/build
    #cmake .. -DCMAKE_BUILD_TYPE=Release || { log "ERROR" "Failed to configure Manifold build"; exit 1; }
    #make -j8 || { log "ERROR" "Failed to build Manifold"; exit 1; }
    #popd

    # Install the python package
    log "INFO" "Installing package"
    pip install -e . || { log "ERROR" "Failed to install the package"; exit 1; }
}

################################################################################
#  Main execution                                                              #  
################################################################################
setup_conda_environment "$1"
install_required_packages
build_package
run_tests
log "INFO" "Done!"
