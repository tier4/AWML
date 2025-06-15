import os

from setuptools import find_packages, setup

setup(
    name="streampetr",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "packaging",
        "ninja==1.11.1.3",
        "fvcore>=0.1.5.post20221221",
        "flash-attn==2.7.3",
        "onnxsim==0.4.36",
    ],
    python_requires=">=3.7",
    author="GitHub: SamratThapa120",
    description="StreamPETR: 3D Object Detection from Streaming Perception",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
)
