.PHONY: default clean coverage _docs docs dtest format lint pages pre-commit test typecheck

PYTEST_CMD = pytest rtdl
VIEW_HTML_CMD = open
DOCS_DIR = docs

default:
	echo "Hello, World!"

clean:
	for x in "bin" "lib" "rtdl"; \
	do \
		find $$x -type f -name "*.py[co]" -delete -o -type d -name __pycache__ -delete; \
	done;
	rm -f .coverage
	rm -rf .ipynb_checkpoints
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf dist
	rm -rf $(DOCS_DIR)/api
	make -C $(DOCS_DIR) clean

coverage:
	coverage run -m $(PYTEST_CMD)
	coverage report -m

_docs:
	make -C $(DOCS_DIR) html

docs: _docs
	$(VIEW_HTML_CMD) $(DOCS_DIR)/build/html/index.html

dtest:
	make -C $(DOCS_DIR) doctest

lint:
	python -m pre_commit_hooks.debug_statement_hook **/*.py
	for x in "bin" "lib" "rtdl"; \
	do \
		isort $$x --check-only; \
		black $$x --check; \
		flake8 $$x; \
	done;

# the order is important: clean must be first, _docs must precede dtest
pre-commit: clean lint test _docs dtest typecheck

test:
	PYTHONPATH='.' $(PYTEST_CMD) $(ARGV)

typecheck:
	mypy rtdl
