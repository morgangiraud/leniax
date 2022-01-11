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
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3.8',
    ],
    keywords=['jax', 'alife', 'lenia', 'leniax'],
    long_description=load_long_description(),
    long_description_content_type='text/markdown',
    packages=find_packages(exclude=['tests', 'tests/*']),
    install_requires=install_requires
)