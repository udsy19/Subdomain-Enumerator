#!/usr/bin/env python3
"""
Ultra-Robust Subdomain Enumerator - Setup Script
For pip installation and distribution
"""

from setuptools import setup, find_packages
import os
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="ultra-robust-subdomain-enumerator",
    version="3.0.0",
    author="Security Research Team",
    author_email="security@example.com",
    description="Advanced AI-Powered Subdomain Enumeration Tool with Beautiful TUI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/ultra-subdomain-enumerator",
    
    # Package configuration
    packages=find_packages(),
    py_modules=["tui_main", "main_tui", "main"],
    
    # Include data files
    package_data={
        "": ["wordlists/*.txt", "*.md", "*.txt"],
    },
    include_package_data=True,
    
    # Requirements
    install_requires=requirements,
    python_requires=">=3.7",
    
    # Entry points
    entry_points={
        "console_scripts": [
            "subdomain-enum=main:main",
            "subdomain-enum-cli=main:main",
            "subdomain-enum-tui=tui_main:main",
        ],
    },
    
    # Classifications
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Information Technology",
        "Intended Audience :: System Administrators",
        "Topic :: Security",
        "Topic :: Internet :: Name Service (DNS)",
        "Topic :: System :: Networking :: Monitoring",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Environment :: Console :: Curses",
    ],
    
    # Keywords
    keywords="subdomain enumeration security dns reconnaissance tui",
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/example/ultra-subdomain-enumerator/issues",
        "Documentation": "https://github.com/example/ultra-subdomain-enumerator/wiki",
        "Source": "https://github.com/example/ultra-subdomain-enumerator",
    },
)