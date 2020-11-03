import shutil
from pathlib import Path

from setuptools import find_packages, setup

extras = {}

setup(
    name="exams-qa",
    version="0.1.0",
    author="Momchil Hardalov, Todor Mihaylov, Dimitrina Zlatkova, Yoan Dinkov, Ivan Koychev, Preslav Nakov",
    description="EXAMS QA",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="Apache",
    url="https://github.com/mhardalov/exams-qa",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        "elasticsearch==7.5.1",
	"transformers==2.8.0",
	"torch==1.6.0",
	"pandas",
	"regex",
	"numpy",
	"more_itertools",
    ],
    extras_require=extras,
    scripts=[],
    python_requires=">=3.7.0",
)
