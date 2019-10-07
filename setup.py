import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dlex",
    version="0.0.1",
    author="Trung V. Dang",
    author_email="trungv.dang@outlook.com",
    description="Deep learning library for research experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/trungd/dlex",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
