.PHONY: setup data train backtest dashboard paper test verify all clean

PYTHON = .venv/Scripts/python
PIP = uv pip

setup:
	uv venv .venv --python 3.10
	$(PIP) install -r requirements.txt --python $(PYTHON)

verify:
	$(PYTHON) scripts/verify.py

data:
	$(PYTHON) scripts/download_data.py --config config/config.yaml

train:
	$(PYTHON) scripts/train_all.py --config config/config.yaml

backtest:
	$(PYTHON) scripts/backtest.py --config config/config.yaml

dashboard:
	$(PYTHON) -m streamlit run src/dashboard/app.py

paper:
	$(PYTHON) scripts/run_paper.py --config config/config.yaml

test:
	$(PYTHON) -m pytest tests/ -v

all: data train backtest test
	@echo "=== Pipeline run complete. Launching dashboard with 'make dashboard' ==="

clean:
	rm -rf __pycache__ .pytest_cache
