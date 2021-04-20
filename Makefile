clean:
	@echo "Remove build, dist, pytest_cache, pycache, and mypy_cache."
	rm -rf build/
	rm -rf dist/

	find . -iname "__pycache__" | xargs rm -rf
	find . -iname "pytest_cache" | xargs rm -rf
	find . -iname ".mypy_cache" | xargs rm -rf

format:
	# TODO: consider autopep8, yapf and others.
	@echo "Formatting Python files with black and isort."
	find . -iname "*.py" | xargs isort
	# isort might change format of black. Recommend to isort first then black.
	find . -iname "*.py" | xargs black

lint:
	@echo "Lint "
	mypy . --ignore-missing-imports
	find . -iname "*.py" | xargs pylint
	flake8
	pydocstyle .

test:
	cd tests
	pytest
	cd -

.PHONY: clean format lint test