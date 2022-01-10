clean:
	@echo "Remove build, dist, pytest_cache, pycache, mypy_cache, and experiment results."
	-rm -rf build/
	-rm -rf dist/
	-rm -rf *.egg-info
	-rm -rf .vscode
	-find . -iname "__pycache__" | xargs rm -rf
	-find . -iname ".pytest_cache" | xargs rm -rf
	-find . -iname ".mypy_cache" | xargs rm -rf
	-find . -iname "2021:*" | xargs rm -rf
	-find ./templates/*  -iname "dataset" | xargs rm -rf

format:
	@-echo "Formatting Python files with black and isort."
	# Disable `isort` for now. This may break order of loading modules.
	# `isort` might change format of `black`. Recommend to `isort` first then `black`.
	# @-find . -iname "*.py" | xargs isort
	# @-find . -iname "*.py" | xargs yapf -i --style='{based_on_style: pep8, indent_width=4}'
	@-find . -iname "*.py" | xargs black --line-length 80

lint:
	@echo "Lint "
	@-mypy . --ignore-missing-imports
	@-find . -iname "*.py" | xargs pylint
	@-flake8
	@-pydocstyle .

test:
	@echo "Test with pytest."
	@cd tests/
	@pytest -vls
	@cd -

doctest:
	@echo "Test with doctest"
	@cd ninpy/
	@find . -iname "*.py" | xargs python -m doctest -v
	@cd -

mypy:
	@echo "Test with mypy."
	@cd ninpy/
	# TODO: use this? find . -iname "*.py" | xargs python -m mypy --ignore-missing-imports
	@find . -iname "*.py" | xargs -n 1 mypy --ignore-missing-imports
	@cd -

pip:
	@echo "Start publish to pip."
	@python setup.py sdist
	@twine upload dist/*

install:
	@python setup.py develop

.PHONY: clean format lint test pip mypy doctest

