.PHONY: help install test lint build

help:
	@echo "Available targets: install, test, lint, build"

install:
	@echo "Install dependencies with pip install -r requirements.txt"

test:
	@echo "Run tests with pytest"

lint:
	@echo "Run linters"

build:
	@echo "Build Docker image"
