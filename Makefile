.PHONY: default clean doctest lint pre-commit typecheck

PACKAGE_ROOT = rtdl

default:
	echo "Hello, World!"

clean:
	find . -type f -name "*.py[co]" -delete -o -type d -name __pycache__ -delete
	rm -rf .ipynb_checkpoints
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf dist

doctest:
	xdoctest $(PACKAGE_ROOT)
	python test_code_blocks.py rtdl/revisiting_models/README.md
	python test_code_blocks.py rtdl/num_embeddings/README.md

lint:
	isort $(PACKAGE_ROOT) --check-only
	black $(PACKAGE_ROOT) --check
	ruff check .

# The order is important.
pre-commit: clean lint doctest typecheck

typecheck:
	mypy $(PACKAGE_ROOT)
