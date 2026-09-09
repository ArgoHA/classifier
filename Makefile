.PHONY: main preprocess split train export bench infer vis demo check_errors test test-fast build

# uv run puts the project venv on sys.path; each module has its own Hydra main().
# Hydra overrides pass straight through: `make train ARGS="model_name=resnet50"`.
PY := uv run python -m img_clf
ARGS ?=

main:
	$(MAKE) train
	$(MAKE) export
	$(MAKE) bench

preprocess:
	$(PY).etl.preprocess $(ARGS)

split:
	$(PY).etl.split $(ARGS)

train:
	$(PY).dl.train $(ARGS)

export:
	$(PY).dl.export $(ARGS)

bench:
	$(PY).dl.bench $(ARGS)

infer:
	$(PY).dl.infer $(ARGS)

demo:
	$(PY).dl.demo $(ARGS)

vis:
	$(PY).dl.vis $(ARGS)

check_errors:
	$(PY).dl.check_errors $(ARGS)

test:
	uv run pytest -q

test-fast:
	uv run pytest -q -m "not slow and not gpu"

build:
	rm -rf dist
	uv build
