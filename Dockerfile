FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
    && grep -v '^torch==' /tmp/requirements.txt > /tmp/requirements-no-torch.txt \
    && pip install --retries 10 --timeout 120 -r /tmp/requirements-no-torch.txt \
    && pip install --retries 10 --timeout 120 --index-url https://download.pytorch.org/whl/cpu torch==2.11.0

COPY app /app

RUN find /app -type f \( -name "*.sh" -o -name "*.ps1" \) -exec sed -i 's/\r$//' {} +

RUN chmod +x \
    /app/init.sh \
    /app/train.sh \
    /app/test.sh \
    /app/run_submission.sh \
    /app/freeze_submission.sh \
    /app/docker_rehearsal.sh \
    /app/data/run.sh

CMD ["/bin/bash", "/app/data/run.sh"]
