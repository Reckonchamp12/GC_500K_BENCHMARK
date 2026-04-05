from setuptools import setup, find_packages

setup(
    name="gc_bench",
    version="1.0.0",
    description="GC-Bench: ML Benchmark for Silicon Grating Coupler Inverse Design",
    author="Ahulray Ray",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "h5py>=3.8.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "xgboost>=1.7.0",
        "lightgbm>=3.3.0",
        "torch>=2.0.0",
        "dtaidistance>=2.3.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
