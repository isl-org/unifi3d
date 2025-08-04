try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import numpy


# Get the numpy include directory.
numpy_include_dir = numpy.get_include()

# Extensions

# triangle hash (efficient mesh intersection)
triangle_hash_module = Extension(
    "utils.libmesh.triangle_hash",
    sources=["utils/libmesh/triangle_hash.pyx"],
    libraries=["m"],  # Unix-like specific
    include_dirs=[numpy_include_dir],
)

# mise (efficient mesh extraction)
mise_module = Extension(
    "utils.libmise.mise",
    sources=["utils/libmise/mise.pyx"],
)

# simplify (efficient mesh simplification)
simplify_mesh_module = Extension(
    "utils.libsimplify.simplify_mesh",
    sources=["utils/libsimplify/simplify_mesh.pyx"],
    include_dirs=[numpy_include_dir],
)

# voxelization (efficient mesh voxelization)
voxelize_module = Extension(
    "utils.libvoxelize.voxelize",
    sources=["utils/libvoxelize/voxelize.pyx"],
    libraries=["m"],  # Unix-like specific
)

# Gather all extension modules
ext_modules = [
    triangle_hash_module,
    mise_module,
    simplify_mesh_module,
    voxelize_module,
]

setup(
    name="triplane_utils",
    version="0.0.1",
    ext_modules=cythonize(ext_modules),
    cmdclass={"build_ext": BuildExtension},
)
