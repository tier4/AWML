import os

from setuptools import find_packages, setup

setup(
    name="streampetr",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "onnx-graphsurgeon==0.5.8",
    ],
    python_requires=">=3.7",
    author="GitHub: SamratThapa120",
)
