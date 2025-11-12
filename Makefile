TEAM ?= NYM
SEASON_START ?= 2025-03-20
OUT ?= build/pdf

report:
	@test -n "$(HITTER)"
	cd src && \
	python generate_report.py \
	  --team $(TEAM) \
	  --hitter "$(HITTER)" \
	  --season_start $(SEASON_START) \
	  --out ../$(OUT)
