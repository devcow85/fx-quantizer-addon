from setuptools import setup, find_packages

setup(
    name="fx-quantizer-addon",
    version="0.0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "torchvision",
        "tqdm",
        "numpy",
        "pandas",
        "dvc",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
        ],
    },
    entry_points={
        "console_scripts": [
            "fx-quantizer-addon=fx_quantizer_addon.main:main",
        ],
    },
)
