FROM ubuntu:latest

RUN apt clean

# Update aptitude with new repo
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git python3 python3-pip nvidia-cuda-toolkit \
    nvidia-cuda-toolkit-gcc \
    gnupg2 \
    curl \
    ca-certificates \
    ssh && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && pip install poetry

RUN mkdir /root/.ssh/ && \
    ssh-keyscan github.com >> /root/.ssh/known_hosts && \
    git clone https://github.com/aai-institute/VeriFlow.git

WORKDIR /VeriFlow
# TODO delete before merging!
RUN git checkout dockerize-experiments
RUN poetry install

# Expose port for TensorBoard
EXPOSE 6006

# TODO: change to start.sh
RUN chmod a+rwx start_test.sh
CMD ["./start_test.sh"]
#CMD ["poetry", "run", "python", "scripts/run-eperiment.py", "--config", "experiments/mnist/mnist.yaml"]
