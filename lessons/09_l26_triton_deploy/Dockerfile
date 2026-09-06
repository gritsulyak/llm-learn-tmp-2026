FROM nvcr.io/nvidia/tritonserver:24.01-py3

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir "numpy==1.26.4" transformers accelerate optimum[onnxruntime]

COPY model_repository /models

EXPOSE 8000 8001 8002

ENTRYPOINT ["tritonserver", "--model-repository=/models"]