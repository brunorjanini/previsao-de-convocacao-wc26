.PHONY: install install-backend install-frontend dev backend frontend stop

# Instala todas as dependências
install: install-backend install-frontend

install-backend:
	pip install -r backend/requirements.txt --break-system-packages

install-frontend:
	cd frontend && npm install

# Roda backend + frontend em paralelo
dev: stop
	@echo "Iniciando backend (porta 8000) e frontend (porta 5173)..."
	@trap 'kill 0' EXIT; \
	python3 -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000 & \
	cd frontend && npm run dev; \
	wait

# Roda só o backend
backend: stop
	python3 -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Roda só o frontend
frontend:
	cd frontend && npm run dev --port 5173

# Para processos nas portas usadas
stop:
	-lsof -ti:8000 | xargs kill -9 2>/dev/null
	-lsof -ti:5173 | xargs kill -9 2>/dev/null
	@echo "Processos encerrados."
