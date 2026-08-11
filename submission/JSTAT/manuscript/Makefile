# Compilation locale, et fabrication de l'archive a televerser sur JSTAT.

LATEX ?= latexmk -pdf -bibtex -interaction=nonstopmode

.PHONY: all check archive clean

all:                ## compile les deux documents, dans le bon ordre
	$(LATEX) manuscript.tex
	$(LATEX) supplement.tex
	$(LATEX) manuscript.tex
	$(LATEX) supplement.tex
	$(LATEX) manuscript.tex
	@$(MAKE) --no-print-directory check

check:              ## echoue si un renvoi reste non resolu
	@for f in manuscript supplement; do \
	  u=$$(grep -ci undefined $$f.log 2>/dev/null; true); \
	  q=$$(pdftotext $$f.pdf - 2>/dev/null | grep -c '??'; true); \
	  p=$$(pdfinfo $$f.pdf 2>/dev/null | awk '/Pages/{print $$2}'); \
	  echo "  $$f : $$p pages, $$u undefined, $$q occurrences de '??'"; \
	  if [ "$$u" != "0" ] || [ "$$q" != "0" ]; then echo "  ECHEC"; exit 1; fi; \
	done
	@echo "  OK"

archive: all        ## produit jstat_upload.tar.gz, pret a televerser
	@rm -rf .archive jstat_upload.tar.gz && mkdir -p .archive/figures
	@cp manuscript.tex refs.bib manuscript.bbl supplement.aux .archive/
	@python3 -c "import re,shutil,pathlib; \
	s=re.sub(r'(?m)^%.*$$','',open('manuscript.tex').read()); \
	[shutil.copy(f'figures/{g}', '.archive/figures/') \
	 for g in re.findall(r'\\\\includegraphics(?:\[[^]]*\])?\{([^}]*)\}', s)]"
	@cd .archive && tar -czf ../jstat_upload.tar.gz *
	@rm -rf .archive
	@echo "  jstat_upload.tar.gz :" $$(du -h jstat_upload.tar.gz | cut -f1)
	@tar -tzf jstat_upload.tar.gz | sed 's/^/    /'

clean:
	latexmk -C manuscript.tex supplement.tex 2>/dev/null || true
	rm -rf *.aux *.blg *.log *.out *.fls *.fdb_latexmk .archive
