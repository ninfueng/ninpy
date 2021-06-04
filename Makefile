clean:
	@echo "Remove build, dist, pytest_cache, pycache, and mypy_cache."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info

	find . -iname "__pycache__" | xargs rm -rf
	find . -iname ".pytest_cache" | xargs rm -rf
	find . -iname ".mypy_cache" | xargs rm -rf

cleanexp:
	@echo "Remove all experiment results"
	find . -iname "2021:*" | xargs rm -rf

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

pytest:
	@echo "Test with pytest."
	cd tests/
	pytest
	cd -

doctest:
	@echo "Test with doctest"
	cd ninpy/
	find . -iname "*.py" | xargs python -m doctest -v
	cd -

mypy:
	@echo "Test with mypy."
	cd ninpy/
	# TODO: use this? find . -iname "*.py" | xargs python -m mypy --ignore-missing-imports
	find . -iname "*.py" | xargs -n 1 mypy --ignore-missing-imports
	cd -

publish:
	@echo "Start publish to pip."
	python setup.py sdist
	twine upload dist/*

.PHONY: clean format lint test publish mypy doctest