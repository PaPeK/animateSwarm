from setuptools import setup, find_packages
setup(
    name = "animateSwarm",
    version = "0.1",
    packages = find_packages(),
    # scripts = ["say_hello.py"],

    # metadata to display on PyPI
    author = "Pascal Klamser",
    author_email = "blechdose@gmail.com",
    description = "animation package for multi-agent systems",
    keywords = "animate, plot, swarm",
    url = "https://github.com/PaPeK/animateSwarm",
    install_requires = [ "matplotlib>=1.5.1", "pathlib>=1.0",
        "scipy>=0.18.1", "numpy>=1.11.0", "h5py>=2.6.0"],
    classifiers = [
        "License :: OSI Approved :: Python Software Foundation License"
    ]
    # could also include long_description, download_url, etc.
)
