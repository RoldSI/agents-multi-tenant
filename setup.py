#!/usr/bin/env python
# coding=utf-8

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="multi-tenant-agents",
    version="0.1.0",
    author="Simon Sure",
    author_email="info@simonsure.com",
    description="A multi-tenant wrapper around smolagents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RoldSI/agents-multi-tenant",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "smolagents>=0.5.0",
        "openai>=1.0.0",
        "python-dotenv>=1.0.0"
    ],
    entry_points={
        "console_scripts": [
            "multi-tenant-example=example:main",
        ],
    },
) 