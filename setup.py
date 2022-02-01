from pkg_resources import DistributionNotFound, get_distribution
from setuptools import setup, find_packages

def load_long_description():
    text = open('README.md').read()
    return text

def get_dist(pkgname):
    try:
        return get_distribution(pkgname)
    except DistributionNotFound:
        return None

install_requires = [
    'ffmpeg-python>=0.2.0', 
    'hilbertcurve>=2.0.5',
    'hydra-core>=1.1.1',
    'jax>=0.2.25', 
    'PyYAML>=6.0',
    'ribs>=0.4.0',
]
# if get_dist('jax') is not None:
#     install_requires.remove('jax>=0.2.25')

extras_require = {
    "all": ["matplotlib>=3.0.0",],
    # Dependencies for examples (NOT tutorials -- tutorial notebooks should
    # install deps with cell magic and only depend on ribs and ribs[all]).
    "viz": [
        "matplotlib>=3.0.0",
        "gym~=0.17.0",  # Strict since different gym may give different results.
        "Box2D~=2.3.10",  # Used in envs such as Lunar Lander.
        "fire>=0.4.0",
        "alive-progress>=1.0.0",

        # Dask
        "dask>=2.0.0",
        "distributed>=2.0.0",
        "bokeh>=2.0.0",  # Dask dashboard.
    ],
    "dev": [
        "pip>=20.3",
        "pylint",
        "yapf",

        # Testing
        "pytest==6.1.2",
        "pytest-cov==2.10.1",
        "pytest-benchmark==3.2.3",
        "pytest-xdist==2.1.0",

        # Documentation
        "Sphinx==3.2.1",
        "sphinx-material==0.0.32",
        "sphinx-autobuild==2020.9.1",
        "sphinx-copybutton==0.3.1",
        "myst-nb==0.10.1",
        "sphinx-toolbox==2.12.1",

        # Distribution
        "bump2version==0.5.11",
        "wheel==0.36.2",
        "twine==1.14.0",
        "check-wheel-contents==0.2.0",
    ]
}

    
setup(
    name="leniax",
    version='1.0.0',
    python_requires='>=3.7.10',  # I keep this minimum python version for Google Collab
    description='Lenia using the JAX library',
    url='https://github.com/stockmouton/leniax',
    author='Morgan Giraud',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        'Programming Language :: Python :: 3.8',
    ],
    keywords=['jax', 'alife', 'lenia', 'leniax'],
    long_description=load_long_description(),
    long_description_content_type='text/markdown',
    packages=find_packages(exclude=['tests', 'tests/*']),
    install_requires=install_requires,
    extras_require=extras_require,
    test_suite="tests",
)