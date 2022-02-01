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
	conda env update --name leniax --file environment_$(OS).yml
	@echo ">>> Conda env exported."
else
	@echo ">>> Please install conda first: brew cask install anaconda"
endif

## Export conda environment
export_env:
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, exporting conda environment."
	conda env export -n leniax | grep -v "^prefix: " > environment_$(OS).yml

	@echo ">>> Conda env exported."
else
	@echo ">>> Please install conda first: brew cask install anaconda"
endif

.PHONY: install export_env


docs: ## generate Sphinx HTML documentation, including API docs
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
.PHONY: docs

##
# CI
###
yapf:
	yapf --style tox.ini -r -i leniax/. tests/. examples/. tools/.

lint:
	flake8 leniax/. tests/. examples/. tools/.

typecheck:
	mypy $(CURRENT_DIR)/leniax $(CURRENT_DIR)/examples

test:
	pytest --disable-pytest-warnings .

test-examples: ## test examples are working
	bash tests/examples.sh

ci: lint typecheck test test-examples

.PHONY: typecheck yapf lint test test-examples ci

###
# Deploy
###
zip:
	python setup.py sdist --format zip

.PHONY: zip


# ffmpeg -r 13 -i %05d.png -c:v libx264 -vf fps=26 -pix_fmt yuv420p beast.mp4