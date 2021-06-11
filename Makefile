.PHONY: better

better:
	isort lib
	isort bin
	black lib
	black bin
	flake8 lib
	flake8 bin
