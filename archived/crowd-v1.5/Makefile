PYTHON ?= python3

test:
	$(PYTHON) -m pytest tests -q

server:
	$(PYTHON) -m uvicorn --factory crowd_v15.server.app:create_app --host 0.0.0.0 --port 8000

client:
	$(PYTHON) -m crowd_v15.client --server http://127.0.0.1:8000 --preset tiny --rounds 1

.PHONY: test server client
