FROM ollama/ollama

WORKDIR /app

RUN apt-get update && apt-get install -y python3 python3-venv python3-pip

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN ollama serve & sleep 10 && ollama pull gemma3:1b

ENTRYPOINT []

CMD ["bash"]