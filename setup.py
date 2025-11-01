from setuptools import setup, find_packages

setup(
    name="echo-adventure",
    version="0.1.0",
    description="Two-layer neural network with trainable inference engine parameters",
    author="cogpy",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
)
