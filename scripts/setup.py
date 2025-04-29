#!/usr/bin/env python3
"""
Setup script for veritas-cli
"""
from setuptools import setup

setup(
    name="veritas-cli",
    version="0.1.0",
    scripts=["cli.py"],
    entry_points={
        "console_scripts": [
            "veritas=cli:main",
        ],
    },
    python_requires=">=3.8",
    description="Command-line interface for Veritas RAG system",
) 