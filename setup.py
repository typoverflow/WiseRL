import os
import pathlib

from setuptools import find_packages, setup


def get_version() -> str:
    init = open(os.path.join("wiserl", "__init__.py"), "r").read().split()
    return init[init.index("__version__") + 2][1:-1]

ROOT_DIR = pathlib.Path(__file__).parent
README = (ROOT_DIR / "README.md").read_text()
VERSION = get_version()

def get_install_requires():
    return [
        "gym>=0.23.1,<=0.24.1",
        "tqdm",
        "numpy",
        "torch",
        "pandas",
        "UtilsRL"
    ]

def get_extras_require():
    return {}

setup(
    name                = "wiserl",
    version             = VERSION,
    description         = "A library desgined for Offline Preference-Based RL algorithms. ",
    long_description    = README,
    long_description_content_type = "text/markdown",
    url                 = "https://github.com/typoverflow/WiseRL",
    author              = "typoverflow",
    author_email        = "typoverflow@outlook.com",
    license             = "MIT",
    packages            = find_packages(),
    include_package_data = True,
    tests_require=["pytest", "mock"],
    python_requires=">=3.7",
    install_requires = [
        "gym>=0.23.1,<=0.24.1",
        "tqdm",
        "numpy",
        "torch",
        "pandas",
        "UtilsRL"
    ]
)
