"""
Setup configuration for Research Assistant Agent.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="research-assistant-agent",
    version="0.1.0",
    author="David Burton",
    author_email="david.burton@example.com",
    description="An AI-powered research assistant for collecting and analyzing academic papers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/davidburton/ResearchAssistantAgent",
    packages=["research_assistant"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.11",
    install_requires=[
        "aiohttp>=3.9.0",
        "click>=8.1.0",
        "rich>=13.5.0",
        "openai>=1.0.0",
        "faiss-cpu>=1.7.4",
        "sentence-transformers>=2.2.0",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "tenacity>=8.2.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "mypy>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "research-assistant=research_assistant.cli:main",
        ],
    },
)