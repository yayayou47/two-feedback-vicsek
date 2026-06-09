PY = ../.venv/bin/python
PKG = flock_simulator
SCRIPTS = $(PKG)/scripts
MS = manuscript

.PHONY: help tests typecheck coverage validate \
        figures manuscript pilot heavy stage3 \
        clean-pdf clean-cache all

help:
	@echo "make tests       -- run the pytest suite (~18 s)"
	@echo "make typecheck   -- mypy strict on the core package"
	@echo "make coverage    -- pytest + coverage report"
	@echo "make validate    -- check bit-for-bit reproducibility"
	@echo "make figures     -- regenerate every figure from data/"
	@echo "make manuscript  -- compile manuscript.pdf via latexmk"
	@echo "make pilot       -- run the 4-mode no-cone pilot (~1 h)"
	@echo "make stage3      -- run the cheap-to-moderate Stage 3 batch (~2 h)"
	@echo "make heavy       -- run the heavy Stage 3 batch (~30 h)"
	@echo "make all         -- tests + typecheck + figures + manuscript"
	@echo "make clean-pdf   -- remove LaTeX build artefacts"
	@echo "make clean-cache -- remove Python/numba caches"

tests:
	$(PY) -m pytest $(PKG)/tests/ -q

typecheck:
	$(PY) -m mypy --config-file mypy.ini \
	    $(PKG)/__init__.py $(PKG)/simulator.py \
	    $(PKG)/core/ $(PKG)/observables/

coverage:
	$(PY) -m pytest $(PKG)/tests/ --cov=$(PKG) -q

validate:
	$(PY) $(SCRIPTS)/validate_reproducibility.py

figures:
	$(PY) -c "import sys; sys.path.insert(0, 'src'); import make_figures; make_figures.main()"

manuscript:
	# Compile both the main file and the separate supplement; run twice
	# so the xr cross-references between them (Figs.~S1/S2 in the main,
	# Eq.~(9) in the supplement) resolve once both .aux files exist.
	cd $(MS) && latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript.tex supplement.tex
	cd $(MS) && latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript.tex supplement.tex

pilot:
	$(PY) $(SCRIPTS)/run_pilot_nocone.py

stage3:
	./$(SCRIPTS)/run_stage3_batch.sh

heavy:
	./$(SCRIPTS)/run_stage3_heavy.sh

all: tests typecheck figures manuscript

clean-pdf:
	cd $(MS) && latexmk -C

clean-cache:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name '.mypy_cache' -exec rm -rf {} +
	find . -type d -name '.pytest_cache' -exec rm -rf {} +
