# Obraz bazowy: sam serwer FastAPI (harness + runtime), CPU.
# Serwer modelu (vLLM) to osobna usluga w docker-compose.yml (profil gpu).
FROM python:3.11-slim

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY experiments ./experiments
COPY configs ./configs
RUN mkdir -p runs

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
