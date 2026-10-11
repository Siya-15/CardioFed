.PHONY: install web web-dev data train federated test serve docker all

install:      ## editable install with dev tools
	pip install -e ".[dev]"

web:          ## build the React UI into frontend/dist
	cd frontend && npm install && npm run build

web-dev:      ## React dev server on :5173 (proxies /api to the backend on :8000)
	cd frontend && npm run dev

data:         ## (re)download the 4 UCI files and rebuild data/processed
	python -m cardiofed.data download
	python -m cardiofed.data clean

train:        ## ablation, tuning, evaluation, figures, saves models/
	python -m cardiofed.train

federated:    ## federated learning simulation -> reports/federated.json
	python -m cardiofed.federated

test:
	pytest

serve:        ## http://localhost:8000
	uvicorn cardiofed.api.main:app --reload

docker:
	docker build -t cardiofed . && docker run --rm -p 8000:8000 cardiofed

all: data train federated test
