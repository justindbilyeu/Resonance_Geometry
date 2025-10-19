from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="resonance-geometry",
    version="0.1.0",
    author="Justin Bilyeu",
    author_email="",  # Fill in if desired
    description="Geometric approaches to information dynamics and AI safety",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/justindbilyeu/Resonance_Geometry",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",  # Or chosen license
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=7.0", "pytest-cov>=4.0", "black>=23.0", "flake8>=6.0"],
        "docs": ["sphinx>=6.0", "sphinx-rtd-theme>=1.0"],
        "llm": ["torch>=2.0", "transformers>=4.30", "datasets>=2.12"],
    },
    entry_points={
        "console_scripts": [
            # Add command-line tools here if desired
            # "rg-simulate=resonance_geometry.cli:main",
        ],
    },
)
