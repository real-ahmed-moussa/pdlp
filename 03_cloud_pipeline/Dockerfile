# Dockerfile
FROM python:3.10.8

# System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential && rm -rf /var/lib/apt/lists/*

# Python Dependenciess
RUN pip install --no-cache-dir \
    tfx==1.15.1 \
    tensorflow==2.15.1 \
    tensorflow-model-analysis==0.46.0 \
    keras-tuner[bayesian]==1.4.7 \
    tensorflow-transform==1.15.0 \
    absl-py==1.4.0 \
    kfp-pipeline-spec==0.1.16

# Copy Pipeline Code
WORKDIR /app
COPY module.py ./

# Default Command (not used by Vertex compile, but handy for local checks)
CMD ["python", "-c", "print('TFX image ready')"]
