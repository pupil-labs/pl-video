.PHONY: install
install: ## Install the virtual environment and install the pre-commit hooks
	@echo "🚀 Creating virtual environment using uv"
	@uv sync
	@uv run pre-commit install

PRETTIER_VERSION := 3.1.0
PRETTIER := npm exec --yes --package=prettier@$(PRETTIER_VERSION) -- prettier
# Checked with Neon X's rules, in ruff.reader.toml and mypy.reader.ini.
READER := src/pupil_labs/video_reader tests/video_reader tools

.PHONY: lint format-check typecheck deptry prettier-format check
lint:
	@echo "🚀 Linting code: Running ruff"
	uv run --locked ruff check --config ruff.reader.toml $(READER)

format-check:
	@echo "🚀 Checking code formatting: Running ruff"
	uv run --locked ruff format --check --config ruff.reader.toml $(READER)

typecheck:
	@echo "🚀 Static type checking: Running mypy"
	uv run --locked mypy
	uv run --locked mypy --config-file mypy.reader.ini

deptry:
	@echo "🚀 Checking for obsolete dependencies: Running deptry"
	uv run --locked deptry .

prettier-format:
	$(PRETTIER) --write .

check: lint format-check typecheck deptry prettier-format ## Run code quality tools
	@echo "🚀 Checking lock file consistency with 'pyproject.toml'"
	@uv lock --locked

.PHONY: compare
compare: ## Compare every frame of pl-video's reader and pupil_labs.video_reader
	uv run python tools/compare_readers.py

.PHONY: benchmark
benchmark: ## Time pl-video's reader against pupil_labs.video_reader
	uv run python tools/benchmark_readers.py

.PHONY: test
test: ## Test the code with pytest
	@echo "🚀 Testing code: Running pytest"
	@uv run python -m pytest --cov --cov-config=pyproject.toml --cov-report=html

.PHONY: build
build: clean-build ## Build wheel file
	@echo "🚀 Creating wheel file"
	@uvx --from build pyproject-build --installer uv

.PHONY: clean-build
clean-build: ## Clean build artifacts
	@echo "🚀 Removing build artifacts"
	@uv run python -c "import shutil; import os; shutil.rmtree('dist') if os.path.exists('dist') else None"


.PHONY: docs-test
docs-test: ## Test if documentation can be built without warnings or errors
	@uv run mkdocs build -s

.PHONY: docs
docs: ## Build and serve the documentation
	@uv run mkdocs serve

.PHONY: help
help:
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help
