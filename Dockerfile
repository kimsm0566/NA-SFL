# Rebuild recipe for a NEW Linux x86_64 server. Do not replace the research container.
ARG BASE_IMAGE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04@sha256:f4d8e1264366940438f0353da6f289c7bef069d993d111f8106086ccd18c4a30
FROM ${BASE_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID
COPY environment/apt-runtime.lock /tmp/apt-runtime.lock
RUN apt-get update \
    && xargs -r apt-get install -y --no-install-recommends < /tmp/apt-runtime.lock \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/* /tmp/apt-runtime.lock
RUN python -m pip install --no-cache-dir --upgrade \
    pip==26.1.2 setuptools==82.0.1 wheel==0.47.0
COPY requirements.txt /opt/na-sfl/requirements.txt
RUN python -m pip install --no-cache-dir -r /opt/na-sfl/requirements.txt
ARG MININET_COMMIT=6eb8973c0bfd13c25c244a3871130c5e36b5fbd7
RUN git init /opt/mininet \
    && git -C /opt/mininet remote add origin https://github.com/mininet/mininet.git \
    && git -C /opt/mininet fetch --depth=1 origin ${MININET_COMMIT} \
    && git -C /opt/mininet checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/mininet rev-parse HEAD)" = "${MININET_COMMIT}" \
    && python -m pip install --no-cache-dir --no-deps --no-build-isolation /opt/mininet \
    && python -m pip check
WORKDIR /workspace/HSFL/src
CMD ["sleep", "infinity"]
