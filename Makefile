clean:
	@echo "Remove build, dist, pytest_cache, pycache, and mypy_cache."
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	find . -type d -name __pycache__ -exec rm -r {} \+
	find . -type d -name .mypy_cache -exec rm -r {} \+

format:
	@echo "Using black and isort."
	isort *.py
	black *.py

