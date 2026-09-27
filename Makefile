NPM ?= npm
CMAKE ?= $(shell command -v cmake 2>/dev/null || echo cmake)
WLEARN_PYTHON ?= python3
PYTHON ?= $(WLEARN_PYTHON)
PIP ?= $(PYTHON) -m pip
WLEARN_CORE_PY ?= ../wlearn/py
WLEARN_POLYGRAD_PY ?=
WLEARN_POLYGRAD_JS ?=
export WLEARN_PYTHON

.PHONY: sync-js-csrc sync-py-csrc build-c test-c test-js test-browser build-py test-py test-cross bench-friedman wheel npm-pack test

sync-js-csrc:
	node js/scripts/sync-csrc.js

sync-py-csrc:
	$(PYTHON) py/scripts/sync-csrc.py

build-c:
	$(CMAKE) -S . -B build -DBUILD_TESTING=ON
	$(CMAKE) --build build

test-c: build-c
	ctest --test-dir build --output-on-failure

test-js:
	cd js && WLEARN_SYM_POLYGRAD_JS=$(WLEARN_POLYGRAD_JS) $(NPM) test

test-browser:
	cd js && $(NPM) run test:browser

build-py:
	cd py && $(PYTHON) -m compileall -q wlearn_sym

test-py: build-py
	PYTHONPATH=py:$(WLEARN_CORE_PY):$(WLEARN_POLYGRAD_PY) $(PYTHON) -m pytest py/tests -q

test-cross: build-py
	PYTHONPATH=py:$(WLEARN_CORE_PY):$(WLEARN_POLYGRAD_PY) $(PYTHON) -m pytest test/test_cross_lang.py -q

bench-friedman:
	node bench/friedman-js.js --output bench/results/friedman-js.latest.json

wheel:
	mkdir -p build/wheelhouse
	cd py && $(PIP) wheel . -w ../build/wheelhouse --no-deps --no-build-isolation

npm-pack:
	cd js && WLEARN_SKIP_BUILD=1 npm_config_cache=/tmp/npm-pack-audit $(NPM) pack --dry-run

test: test-js test-py test-cross
