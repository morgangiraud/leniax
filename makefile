# Default
all: test
.PHONY: all

OS := $(shell uname | tr '[:upper:]' '[:lower:]')
CURRENT_DIR=$(shell pwd)

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

###
# Package
###
install:
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, creating conda environment."
	conda env create -f environment_$(OS).yml
	@echo ">>> Conda env created."
else
	@echo ">>> Please install conda first: brew cask install anaconda"
endif

## Export conda environment
update_env:
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, exporting conda environment."
	conda env update --name lenia-nft --file environment_$(OS).yml
	@echo ">>> Conda env exported."
else
	@echo ">>> Please install conda first: brew cask install anaconda"
endif

## Export conda environment
export_env:
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, exporting conda environment."
	conda env export -n lenia-nft | grep -v "^prefix: " > environment_$(OS).yml

	@echo ">>> Conda env exported."
else
	@echo ">>> Please install conda first: brew cask install anaconda"
endif

.PHONY: install export_env

##
# CI
###
lint:
	flake8 lenia/. tests/. scripts/.

yapf:
	yapf --style tox.ini -r -i lenia/. tests/. scripts/.

test:
	pytest .

ci: lint typecheck test

.PHONY: typecheck yapf lint test ci

###
# Deploy
###
zip:
	python setup.py sdist --format zip

.PHONY: zip