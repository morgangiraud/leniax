from setuptools import setup, find_packages


def load_long_description():
    text = open('README.md').read()
    return text

setup(
    name="leniax",
    version='1.0',
    python_requires='>=3.7.10',
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
    install_requires=[
        'hydra-core>=1.1.1',
        'ribs>=0.4.0',
        'ffmpeg-python>=0.2.0', 
        'jax>=0.2.26', 
        'jaxlib>=0.1.75',
        'PyYAML>=0.2.5'
    ]
)