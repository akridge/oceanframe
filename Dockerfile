FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOST=0.0.0.0 \
    APP_PORT=80 \
    OPEN_BROWSER=0 \
    LIB_DATA_DIR=/data/library

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Build with --build-arg WITH_ML=1 to bake in torch, CLIP and Ultralytics.
# It is off by default because it adds ~2.5 GB to the image, and the library is
# fully functional without it (see requirements-ml.txt).
ARG WITH_ML=0
RUN if [ "$WITH_ML" = "1" ]; then pip install --no-cache-dir -r requirements-ml.txt; fi

COPY . .

RUN mkdir -p /app/uploads/thumbs /data/library

EXPOSE 80

CMD ["python", "launch.py"]
