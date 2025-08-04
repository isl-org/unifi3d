import platform
import pytest
import torch
import torch_geometric
import hydra
import subprocess
from pathlib import Path


def test_torch_installed() -> None:
    """Test if PyTorch is installed."""
    try:
        _ = torch.__version__
    except AttributeError:
        pytest.fail("PyTorch does not appear to be properly installed.")


def test_torch_geometric_installed() -> None:
    """Test if PyTorch Geometric is installed."""
    try:
        _ = torch_geometric.__version__
    except AttributeError:
        pytest.fail("PyTorch Geometric does not appear to be properly installed.")


def test_hydra_installed() -> None:
    """Test if Hydra is installed."""
    try:
        _ = hydra.__version__
    except AttributeError:
        pytest.fail("Hydra does not appear to be properly installed.")


def test_gpu_available() -> None:
    """
    Test if at least one GPU device is detected based on the operating system.

    - On Linux: Checks for CUDA availability.
    - On macOS (Apple Silicon): Checks for MPS (Metal Performance Shaders) availability.

    Raises:
        AssertionError: If no GPU device is detected for the respective backend.
        NotImplementedError: If the operating system is neither Linux nor macOS.
    """
    os_type = platform.system()
    if os_type == "Darwin":
        # macOS (Apple Silicon) typically uses MPS for GPU acceleration
        if torch.backends.mps.is_available():
            print(
                "[INFO] MPS (Metal Performance Shaders) is available for AI acceleration on macOS."
            )
        else:
            raise AssertionError(
                "No MPS devices detected on macOS. Ensure that you're using a compatible PyTorch build with MPS support."
            )
    elif os_type == "Linux" or os_type == "Windows":
        if torch.cuda.is_available():  # API for Nvidia GPU support
            print("[INFO] CUDA is available for AI acceleration on Linux.")
        elif torch.xpu.is_available():  # API for Intel GPU support
            print("[INFO] Intel XPU is available for AI acceleration on Linux.")
        else:
            raise AssertionError(
                "No CUDA devices detected on Linux. Ensure that CUDA is properly installed and configured."
            )
    else:
        raise NotImplementedError(
            f"GPU availability check not implemented for OS: {os_type}"
        )


def test_gpu_usable() -> None:
    """
    Test if the detected GPU device is usable based on the operating system.

    - On Linux: Checks CUDA GPU usability.
    - On macOS (Apple Silicon): Checks MPS (Metal Performance Shaders) GPU usability.

    Raises:
        pytest.fail: If the GPU is detected but not usable.
        pytest.skip: If no suitable GPU is available or the OS is unsupported.
    """
    os_type = platform.system()
    if os_type == "Darwin":
        # macOS (Apple Silicon) typically uses MPS for GPU acceleration
        if torch.backends.mps.is_available():
            try:
                # Attempt to perform a simple tensor operation on MPS
                _ = torch.tensor([1.0], device="mps")
                print("[INFO] MPS GPU is available and usable.")
            except RuntimeError as e:
                pytest.fail(f"MPS GPU detected but not usable. Exception: {e}")
        else:
            pytest.skip("No MPS devices available to test.")
    elif os_type == "Linux" or os_type == "Windows":
        # Linux systems typically use CUDA for GPU acceleration
        if torch.cuda.is_available():
            try:
                # Attempt to perform a simple tensor operation on CUDA
                _ = torch.tensor([1.0], device="cuda")
                print("[INFO] CUDA GPU is available and usable.")
            except RuntimeError as e:
                pytest.fail(f"CUDA GPU detected but not usable. Exception: {e}")
        elif torch.xpu.is_available():
            try:
                # Attempt to perform a simple tensor operation on Intel XPU
                _ = torch.tensor([1.0], device="xpu")
                print("[INFO] Intel XPU is available and usable.")
            except RuntimeError as e:
                pytest.fail(f"Intel XPU detected but not usable. Exception: {e}")
        else:
            pytest.skip("No GPU/XPU devices available to test.")
    else:
        pytest.skip(f"GPU usability test not implemented for OS: {os_type}")


# ---------------------------------------------------------------------------
# Manifold tests
# ---------------------------------------------------------------------------


# def _get_manifold_exec_path() -> Path:
#     """
#     Compute the path to the Manifold executable built in
#     core/utils/Manifold/build, taking the .exe suffix into account on Windows.
#     """
#     root = Path(__file__).resolve().parents[1]
#     exec_name = "manifold.exe" if platform.system() == "Windows" else "manifold"
#     return root / "core" / "utils" / "Manifold" / "build" / exec_name


# def test_manifold_executable_present() -> None:
#     """
#     Ensure that the Manifold executable exists after compilation
#     """
#     exec_path = _get_manifold_exec_path()
#     assert exec_path.is_file(), f"Manifold executable not found at {exec_path}"


# def test_manifold_help_runs() -> None:
#     """
#     Run `manifold --help` to verify the binary starts successfully.
#     """
#     exec_path = _get_manifold_exec_path()
#     if not exec_path.is_file():
#         pytest.skip("Manifold executable not built; skipping runtime test.")
#     result = subprocess.run(
#         [str(exec_path), "--help"],
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE,
#         text=True,
#     )
#     assert (
#         result.returncode == 0
#     ), f"'manifold --help' failed with exit code {result.returncode}.\nSTDERR:\n{result.stderr}"
