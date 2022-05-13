default:
  just --list

# clean all build, python, and lint files
clean:
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -fr {} +
	find . -name '*.pyc' -exec rm -fr {} +
	find . -name '*.pyo' -exec rm -fr {} +
	find . -name '__pycache__' -exec rm -fr {} +
	find . -name '.tox' -exec rm -fr {} +
	find . -name '.mypy_cache' -exec rm -fr {} +

# clean all stored or generated data
clean-data:
	find . -name 'upload-manifest.jsonl' -exec rm -fr {} +
	find . -name 'training-data' -exec rm -fr {} +
	find . -name 'prepared-*-dataset' -exec rm -fr {} +

# clean all build, python, lint files, and stored and generated data
clean-all:
	just clean clean-data

# lint, format, and check all files
lint:
	tox -e lint