import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hawkins-machine-learning",
    version="0.0.1",
    author="Jose Peña Seco",
    author_email="josericardopenase@gmail.com",
    description="A neural network micro-library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/josericardopenase/hawkins",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)