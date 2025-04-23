import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):

    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):

    def build_extension(self, ext: CMakeExtension) -> None:
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DRAFT_NVTX=OFF",
            f"-DCMAKE_CUDA_ARCHITECTURES='NATIVE'",
            f"-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        ]
        build_args = []

        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            if hasattr(self, "parallel") and self.parallel:
                build_args += [f"-j{self.parallel}"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        subprocess.run(["cmake", ext.sourcedir, *cmake_args],
                       cwd=build_temp,
                       check=True)
        subprocess.run(["cmake", "--build", ".", *build_args],
                       cwd=build_temp,
                       check=True)


setup(
    name="F2CGT",
    version="0.0.1",
    author="",
    author_email="",
    description="",
    long_description="",
    packages=find_packages(),
    ext_modules=[CMakeExtension("F2CGTLib", "..")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    )

