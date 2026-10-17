.PHONY: test data eval clean format

test:
	pytest tests/ -v

data:
	python -m vericall.data.prep

eval:
	python -m vericall.eval.evaluate

render:
	python scripts/render_results.py

format:
	ruff format vericall tests scripts
	ruff check --fix vericall tests scripts

clean:
	rm -rf __pycache__ .pytest_cache build dist *.egg-info
