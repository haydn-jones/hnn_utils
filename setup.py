from setuptools import setup, find_packages

setup(
    name="hnn_utils",
    version="0.1.8",
    packages=find_packages(),
    install_requires=[
        "torch",
        "rotary-embedding-torch",
        "lightning",
    ],
    author="Haydn Jones",
    author_email="haydnjonest@gmail.com",
    description="Various utilities used throughout my research",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/haydn-jones/hnn_utils",
)