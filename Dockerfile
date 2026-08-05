FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install --no-cache-dir .
ENTRYPOINT ["epc-train"]

