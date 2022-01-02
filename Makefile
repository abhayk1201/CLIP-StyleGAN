# Define macros
UNAME_S := $(shell uname -s)
PYTHON := python

## HYDRA_FLAGS : set as -m for multirun
SEED := 0

.PHONY: help docs
.DEFAULT: help

## install.mamba: Mamba package manager
install.mamba:
	conda install -y mamba -c conda-forge

## install.cpu: CPU mode
install.cpu: install.mamba
	conda activate
	mamba env update -n clip -f environment.yml --prune

## install.gpu: Additional CUDA dependencies
install.gpu: install.mamba
	mamba env update -n clip -f environment_gpu.yml --prune

help : Makefile
    ifeq ($(UNAME_S),Linux)
		@sed -ns -e '$$a\\' -e 's/^##//p' $^
    endif
    ifeq ($(UNAME_S),Darwin)
        ifneq (, $(shell which gsed))
			@gsed -sn -e 's/^##//p' -e '$$a\\' $^
        else
			@sed -n 's/^##//p' $^
        endif
    endif
