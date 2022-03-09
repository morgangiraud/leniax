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

extras_require = {
    "all": [
        "matplotlib>=3.0.0",
        "tensorflow>=2.7.0",
        "flax>=0.4.0",
    ],
    "optim": [
       "tensorflow>=2.7.0",
       "flax>=0.4.0",
    ],
}

    
setup(
    name="leniax",
    version='0.2.0',
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