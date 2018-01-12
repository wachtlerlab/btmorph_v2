from setuptools import setup

setup(
    name="btmorph2",
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    packages=['btmorph2'],
    install_requires=[
        "numpy>=1.11.2",
        "matplotlib>=1.5.3",
        "scipy>=0.18.1",
        "pandas>=0.22",
        "openpyxl>=2.4.9",
        "xlrd>=1.1",
        "networkX==1.11",
        "pathlib2>=2.2.1",
        "future>=0.16"
     ],
    python_requires=">=2.7",
)
