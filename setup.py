from setuptools import setup, find_packages

setup(
    name="qtemp-fet-analyzer",
    version="4.0.0",
    author="QTEMP Team",
    description="Dual-sweep FET analyzer",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "streamlit>=1.28.0",
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
)
