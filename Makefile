# The performance campaign of the thesis (ADR-0006), one target per step, so
# the same commands run on every machine and the lab note is `make bench-all
# MACHINE=<id>`. Every target is a thin wrapper around `uv run unproject`;
# nothing here deletes, moves or edits anything outside the artifacts root it
# is given, and the artifacts root defaults to a directory that is gitignored.
#
#   make help                       what is here
#   make bench-env                  the environment, from the lockfile
#   make bench-datasets FROM=<dir> MUPOTS=accept  the campaign's footage, verified by digest
#   make bench-preflight MACHINE=B  is this machine ready? (read-only)
#   make bench-all MACHINE=B        matrix, people and CPU suites, each preflighted
#   make bench-collect MACHINE=B    one archive of the results, to carry off the machine
#   make bench-campaign             the cross-machine table, from every machine's results
#
# MACHINE is an id from benchmarks/machines.toml and is required by anything
# that measures. CAMPAIGN is the root every machine writes under, one
# directory per machine, and the same on every machine so the archives fit
# together when they are brought back.

SHELL := /bin/bash

MACHINE ?=
CAMPAIGN ?= artifacts/campaign
ROOT := $(CAMPAIGN)/$(MACHINE)

SUITES := experiments/benchmark/v2
MATRIX := $(SUITES)/matrix-suite.toml
PEOPLE := $(SUITES)/people-suite.toml
CPU := $(SUITES)/cpu-suite.toml

UV := uv run --frozen
# On macOS, keep the machine awake for the hour a suite takes; elsewhere a no-op.
CAFFEINATE := $(shell command -v caffeinate 2>/dev/null)
AWAKE := $(if $(CAFFEINATE),$(CAFFEINATE) -i,)
STAMP := $(shell date -u +%Y%m%dT%H%M%SZ)

.PHONY: help bench-env bench-datasets bench-preflight bench-matrix bench-people bench-cpu \
        bench-all bench-summary bench-collect bench-campaign _require-machine

help:
	@sed -n '1,19p' $(MAKEFILE_LIST) | sed 's/^# \{0,1\}//'

_require-machine:
	@test -n "$(MACHINE)" || { echo "MACHINE=<id> is required; ids are in benchmarks/machines.toml" >&2; exit 2; }

## Environment: the lockfile's versions and nothing else. `--frozen` refuses
## to resolve anew, so every machine gets the same torch, numpy and estimator.
bench-env:
	uv sync --frozen --extra multihmr
	$(UV) python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'mps', torch.backends.mps.is_available())"

## Footage: the campaign's bytes, from a directory (FROM=...) or a deposit
## (DEPOSIT=https://...), plus MOT16 from MOTChallenge and, once you have read
## its terms and say MUPOTS=accept, MuPoTS-3D TS10/TS11. Verifies every digest.
bench-datasets:
	benchmarks/fetch-datasets.sh $(if $(FROM),--from-dir "$(FROM)",) $(if $(DEPOSIT),--from-deposit "$(DEPOSIT)",) $(if $(filter accept,$(MUPOTS)),--accept-mupots-terms,)

## Readiness, read-only apart from one report beside the results. Fetches the
## pinned weights if absent, which the first job would do anyway.
bench-preflight: _require-machine
	$(UV) unproject benchmark-preflight $(MATRIX) --machine $(MACHINE) --artifacts-root $(ROOT) --fetch-weights
	$(UV) unproject benchmark-preflight $(PEOPLE) --machine $(MACHINE) --artifacts-root $(ROOT)
	$(UV) unproject benchmark-preflight $(CPU) --machine $(MACHINE) --artifacts-root $(ROOT)

## The suites. Each is preflighted first and refused if the preflight fails;
## each ends with its own summary, derived from the raw results.
bench-matrix: _require-machine
	$(UV) unproject benchmark-preflight $(MATRIX) --machine $(MACHINE) --artifacts-root $(ROOT) --fetch-weights
	$(AWAKE) $(UV) unproject benchmark-suite $(MATRIX) --machine $(MACHINE) --artifacts-root $(ROOT)
	$(UV) unproject benchmark-suite-summary $(MATRIX) --artifacts-root $(ROOT)

bench-people: _require-machine
	$(UV) unproject benchmark-preflight $(PEOPLE) --machine $(MACHINE) --artifacts-root $(ROOT) --fetch-weights
	$(AWAKE) $(UV) unproject benchmark-suite $(PEOPLE) --machine $(MACHINE) --artifacts-root $(ROOT)
	$(UV) unproject benchmark-suite-summary $(PEOPLE) --artifacts-root $(ROOT)

bench-cpu: _require-machine
	$(UV) unproject benchmark-preflight $(CPU) --machine $(MACHINE) --artifacts-root $(ROOT) --fetch-weights
	$(AWAKE) $(UV) unproject benchmark-suite $(CPU) --machine $(MACHINE) --artifacts-root $(ROOT)
	$(UV) unproject benchmark-suite-summary $(CPU) --artifacts-root $(ROOT)

bench-all: bench-matrix bench-people bench-cpu

## Regenerate the summaries from the raw results; no inference.
bench-summary: _require-machine
	@for suite in $(MATRIX) $(PEOPLE) $(CPU); do \
	    $(UV) unproject benchmark-suite-summary $$suite --artifacts-root $(ROOT) || true; \
	done

## Everything the thesis needs from this machine, as one archive: the job
## results, run records, preflight reports and summaries. Not the ingest
## frames, which regenerate from the footage. Written beside the machine's
## directory; nothing is removed.
bench-collect: _require-machine
	@test -d "$(ROOT)/benchmark-v2" || { echo "no results under $(ROOT)/benchmark-v2" >&2; exit 1; }
	tar -czf "$(CAMPAIGN)/$(MACHINE)-$(STAMP).tar.gz" -C "$(CAMPAIGN)" "$(MACHINE)/benchmark-v2"
	@echo "wrote $(CAMPAIGN)/$(MACHINE)-$(STAMP).tar.gz"
	@shasum -a 256 "$(CAMPAIGN)/$(MACHINE)-$(STAMP).tar.gz"

## The cross-machine table, from every <machine>/benchmark-v2/<suite>/summary.json
## under the campaign root. Run where the archives were unpacked.
bench-campaign:
	$(UV) unproject benchmark-campaign-summary $(CAMPAIGN)
