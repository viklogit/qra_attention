from setuptools import setup, find_packages

setup(
    name="qra_attention",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
    python_requires=">=3.8",
    author="Victor Martinez",
    description="Quantum-Ready Attention: Kernel-based attention mechanisms for Transformers",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
)