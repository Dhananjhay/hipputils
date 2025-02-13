from setuptools import setup

long_description = __doc__.strip()  # Remove leading and trailing newlines

setup(
    name="hipputils",
    version="0.0.1",  # see semantic versioning (<https://semver.org/spec/v2.0.0.html>)
    description="Hippunfold Python Utility function and classes",
    long_description=long_description,
    url="https://github.com/Dhananjhay/hipputils",
    author="Hippunfold Team",
    # author_email="dhananjhay03@gmail.com",
    # maintainer="Dhananjhay Bansal",
    # maintainer_email="dhananjhay03@gmail.com"
    packages=[
        "hipputils",
    ],
    install_requires=[
        "numpy",
        "nibabel",
        "nilearn",
        "pyvista",
        "pygeodesic",
        "matplotlib",
        "scipy",
        "lib",
        "pandas",
        "seaborn",
    ],
    license="MIT",
    python_requires=">=3.9",
    platforms=["Linux"],
)