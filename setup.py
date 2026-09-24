"""
Project: Learning Multi-modal Representations by Watching Hundreds of Surgical Video Lectures
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
"""
import os
from setuptools import setup, find_packages

def read_requirements(path):
    reqs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            reqs.append(line)
    return reqs

setup(
    name="surgvlp",
    version="0.1.1",
    description="",
    author="CAMMA",
    packages=find_packages(exclude=["tests*"]),
    install_requires=read_requirements(os.path.join(os.path.dirname(__file__), "requirements.txt")),
    include_package_data=True,
    extras_require={'dev': ['pytest']},
)
