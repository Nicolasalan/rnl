.PHONY: sim
sim:
	@uv run python -m main sim

.PHONY: learn
learn:
	@uv run python -m main learn

.PHONY: publish
publish:
	@uv publish --build -u __token__ -p $(RNL_PYPI_TOKEN)

.PHONY: install
install:
	@uv sync

uv     = uv
BLACK  = $(uv) run black
ISORT  = $(uv) run isort
RUFF   = $(uv) run ruff
SRC_DIR= ./

.PHONY: lint
lint:
	$(ISORT) $(SRC_DIR)
	$(BLACK) $(SRC_DIR)
	$(RUFF) check $(SRC_DIR)
