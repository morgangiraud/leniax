# Installation

## From source
To install a version from source, clone the repo
```
git clone https://github.com/morgangiraud/leniax
cd leniax
```

Install Leniax library with conda (make sure you have it before typing the following command): `make install`

Then activate the environment: `conda activate leniax`

Finally, install the lib itself: `pip install .`

## Verification
You can make sure that everything is working fine by running the following command: `make ci`