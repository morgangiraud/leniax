from setuptools import setup, find_packages


def load_long_description():
    text = open('README.md', encoding='utf-8').read()
    return text

setup(
    name="JAX lenia",
    version='0.1',
    python_requires='>=3.9',
    description='Lenia using the JAX library',
    url='https://github.com/stockmouton/lenia',
    author='Morgan Giraud',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3.9',
    ],
    keywords=['jax', 'alife', 'lenia'],
    long_description=load_long_description(),
    long_description_content_type='text/markdown',
    packages=find_packages(exclude=['tests', 'tests/*']),
    install_requires=[]
)