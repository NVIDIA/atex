# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import atexit
import copy
import io
import os
import re
import subprocess
import tempfile
from distutils.version import LooseVersion
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

__version__ = '0.0.2'
REQUIRED_PACKAGES = [
    'tensorflow >= 2.1.0',
]
path = os.path.dirname(os.path.realpath(__file__))

class CMakeExtension(Extension):
    def __init__(self, name, cmake_path, sources, **kwargs):
        super(CMakeExtension, self).__init__(name, sources=sources, **kwargs)
        self.cmake_path = cmake_path

ext_modules = []
ext_modules.append(
    CMakeExtension(
        name="structured_sparsity",
        cmake_path=os.path.join(path, "atex", "structured_sparsity"),
        sources=[],
    )
)
ext_modules.append(
    CMakeExtension(
        name="nv_norms",
        cmake_path=os.path.join(path, "atex", "nv_norms"),
        sources=[],
    )
)

def get_cmake_bin():
    cmake_bin = "cmake"
    try:
        out = subprocess.check_output([cmake_bin, "--version"])
    except OSError:
        cmake_installed_version = LooseVersion("0.0")
    else:
        cmake_installed_version = LooseVersion(
            re.search(r"version\s*([\d.]+)", out.decode()).group(1)
        )

    if cmake_installed_version < LooseVersion("3.18.0"):
        print(
            "Could not find a recent CMake to build Transformer Engine. "
            "Attempting to install CMake 3.18 to a temporary location via pip.",
            flush=True,
        )
        cmake_temp_dir = tempfile.TemporaryDirectory(prefix="nvte-cmake-tmp")
        atexit.register(cmake_temp_dir.cleanup)
        try:
            _ = subprocess.check_output(
                ["pip", "install", "--target", cmake_temp_dir.name, "cmake~=3.18.0"]
            )
        except Exception:
            raise RuntimeError(
                "Failed to install temporary CMake. "
                "Please update your CMake to 3.18+."
            )
        cmake_bin = os.path.join(cmake_temp_dir.name, "bin", "run_cmake")
        with io.open(cmake_bin, "w") as f_run_cmake:
            f_run_cmake.write(
                f"#!/bin/sh\nPYTHONPATH={cmake_temp_dir.name} {os.path.join(cmake_temp_dir.name, 'bin', 'cmake')} \"$@\""
            )
        os.chmod(cmake_bin, 0o755)

    return cmake_bin

class CMakeBuildExtension(build_ext, object):
    def __init__(self, *args, **kwargs) -> None:
        super(CMakeBuildExtension, self).__init__(*args, **kwargs)

    def build_extensions(self) -> None:
        print("Building CMake extensions!")
        self.cmake_bin = get_cmake_bin()
        for extension in self.extensions:
            self.build_cmake(extension)
    
    def build_cmake(self, extension) -> None:
        config = "Debug" if self.debug else "Release"

        ext_name = extension.name
        build_dir = self.get_ext_fullpath(ext_name).replace(
            self.get_ext_filename(ext_name), ""
        )
        build_dir = os.path.abspath(build_dir)

        cmake_args = [
            "-DCMAKE_BUILD_TYPE=" + config,
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(config.upper(), build_dir),
        ]
        try:
            import ninja
        except ImportError:
            pass
        else:
            cmake_args.append("-GNinja")

        cmake_args = cmake_args

        cmake_build_args = ["--config", config]

        cmake_build_dir = os.path.join(self.build_temp, ext_name, config)
        if not os.path.exists(cmake_build_dir):
            os.makedirs(cmake_build_dir)

        config_and_build_commands = [
            [self.cmake_bin, extension.cmake_path] + cmake_args,
            [self.cmake_bin, "--build", "."] + cmake_build_args,
        ]

        print(f"Running CMake in {cmake_build_dir}:")
        for command in config_and_build_commands:
            print(" ".join(command))
        sys.stdout.flush()

        # Config and build the extension
        try:
            for command in config_and_build_commands:
                subprocess.check_call(command, cwd=cmake_build_dir)
        except OSError as e:
            raise RuntimeError("CMake failed: {}".format(str(e)))

setup(
    name="atex",
    version=__version__,
    packages=find_packages(),
    description=('tensorflow-nv-norms is for fused layer/instance normalization ops for TensorFlow'),
    ext_modules=ext_modules,
    cmdclass={"build_ext": CMakeBuildExtension},
    author='NVIDIA',
    author_email='kaixih@nvidia.com',
    install_requires=REQUIRED_PACKAGES,
    # PyPI package information.
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
    ],
    license='Apache 2.0',
    keywords='tensorflow custom op machine learning',
)
