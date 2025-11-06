import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="photolooper",
    version="0.2.0",
    author="Kevin Maik Jablonka, Michael Ringleb",
    author_email="mail@kjablonka.com",
    description="A package to control photocat HTE with separated photosensitizer and catalyst",
    url="https://github.com/MichaelRingleb/photocat-hte-prescreen",
    packages=setuptools.find_packages(where="src"),
    install_requires=["rd6006", "pyrolib"],
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
