# syntax=docker/dockerfile:1
# =============================================================================
# vLLM main from source + MoRI from source + vllm-router.
#
# Base is upstream's published Dockerfile.rocm_72_base output (the ROCm 7.2
# image; ROCm 10 became upstream's default on 17 Oct, #58761), which supplies
# ROCm, torch, triton, AITER and FlyDSL (AITER pins flydsl==0.4.0) exactly as
# vLLM main pins them. On top of it this file builds, in order:
#   1. MoRI          from source at MORI_REF
#   2. vLLM          from source at VLLM_REF (basemam/vllm main), using the
#                    same commands as upstream docker/Dockerfile.rocm
#   3. vllm-router   from source at ROUTER_REF
#
# Pins (from docker/Dockerfile.rocm_72_base / Dockerfile.rocm_72 at VLLM_REF):
#   ROCm 7.2.3, torch ROCm/pytorch 733fca1 (release/2.13),
#   triton ROCm/triton 669b31a, AITER v0.1.26, flydsl 0.4.0  <- BASE_IMAGE
#   MoRI v1.2.4 (upstream still pins v1.2.3.post1), triton_kernels 669b31ac
#                                                            <- built here
#
# The MoRI combine fix is already upstream (vllm#56073). The MoRI dispatch trim
# and the SWA bounded_replay fix are commits on VLLM_REF, not boot patches.
#
# Build (compute node only, never login/WSL):
#   docker build --ulimit nofile=1048576:1048576 \
#     -f mad/docker/vllm_main.dsv4.ubuntu.amd.Dockerfile -t <tag> mad/docker
# or: cd mad/scripts/vllm_dissag && sbatch --qos=low build_vllm_main_image.slurm
# =============================================================================
ARG BASE_IMAGE=vllm/vllm-openai-rocm:base-nightly-rocm72-83c078d955cd6e0c63d3b12abe669c8de936a370
FROM ${BASE_IMAGE}
ARG BASE_IMAGE

ARG GFX_ARCH=gfx942
ARG MAX_JOBS=64
ENV PYTORCH_ROCM_ARCH=${GFX_ARCH} \
    MAX_JOBS=${MAX_JOBS}

# Compute nodes cannot reach the Ubuntu mirrors over plain HTTP (475672); HTTPS works.
RUN sed -i -E 's#http://(archive|security)\.ubuntu\.com#https://\1.ubuntu.com#g' \
        /etc/apt/sources.list $(ls /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources 2>/dev/null) && \
    apt-get update -q -y && apt-get install -q -y \
        sqlite3 libsqlite3-dev libfmt-dev libmsgpack-dev libsuitesparse-dev \
        apt-transport-https ca-certificates wget curl gnupg git \
        libnuma-dev ccache mold \
        protobuf-compiler libprotobuf-dev libgrpc++-dev protobuf-compiler-grpc && \
    rm -rf /var/lib/apt/lists/* && \
    (command -v uv >/dev/null || python3 -m pip install -q uv) && \
    mkdir -p /app && \
    echo "BASE_IMAGE=${BASE_IMAGE}" >> /app/versions.txt && \
    python3 -c 'import torch; print(f"TORCH={torch.__version__} HIP={torch.version.hip}")' >> /app/versions.txt && \
    test -d "$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')/aiter" && \
    pip list 2>/dev/null | grep -iE '^(amd[-_])?aiter |^flydsl |^mori |^triton |^torch ' | sed 's/^/BASE_PKG /' >> /app/versions.txt

# The base carries upstream CI's sccache (S3-backed). vLLM's setup.py uses
# sccache whenever it is on PATH, and without AWS credentials it cannot start,
# so remove it as upstream's own final_common stage does.
RUN rm -f /usr/bin/sccache /usr/local/bin/sccache && rm -rf /opt/sccache-wrappers && \
    ! command -v sccache
ENV SCCACHE_BUCKET= \
    SCCACHE_REGION= \
    SCCACHE_ENDPOINT= \
    SCCACHE_S3_NO_CREDENTIALS= \
    SCCACHE_IDLE_TIMEOUT=

# -----------------------------------------------------------------------------
# 1. MoRI from source. NIC vendor libs and proxy deps as in upstream mori_base.
# -----------------------------------------------------------------------------
ARG MORI_REPO=https://github.com/ROCm/mori.git
ARG MORI_REF=v1.2.4
ARG NIC_BACKEND=all
ARG AINIC_VERSION=1.117.5-a-77
ARG UBUNTU_CODENAME=jammy
RUN /bin/bash -lc 'set -euo pipefail; \
    install_ainic() { \
      mkdir -p /etc/apt/keyrings; \
      curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor > /etc/apt/keyrings/amdainic.gpg; \
      echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/amdainic.gpg] https://repo.radeon.com/amdainic/pensando/ubuntu/${AINIC_VERSION} ${UBUNTU_CODENAME} main" \
        > /etc/apt/sources.list.d/amdainic.list; \
      apt-get update && apt-get install -y --no-install-recommends libionic-dev ionic-common; \
      rm -rf /var/lib/apt/lists/*; \
    }; \
    install_bnxt() { \
      install -m 0755 -d /etc/apt/keyrings; \
      curl -fsSL https://packages.broadcom.com/artifactory/api/security/keypair/PackagesKey/public \
        -o /etc/apt/keyrings/broadcom-nic.asc; \
      chmod a+r /etc/apt/keyrings/broadcom-nic.asc; \
      echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/broadcom-nic.asc] https://packages.broadcom.com/artifactory/ethernet-nic-debian-public jammy main" \
        > /etc/apt/sources.list.d/broadcom-nic.list; \
      apt-get update && apt-get install -y --no-install-recommends bnxt-rocelib=235.2.86.0; \
      cp -a /usr/local/lib/x86_64-linux-gnu/libbnxt_re* /usr/local/lib/; \
      ldconfig; \
      rm -rf /var/lib/apt/lists/*; \
    }; \
    case "${NIC_BACKEND}" in \
      none) ;; all) install_ainic; install_bnxt ;; ainic) install_ainic ;; bnxt) install_bnxt ;; \
      *) echo "ERROR: NIC_BACKEND=${NIC_BACKEND}"; exit 2 ;; \
    esac; \
    pip install --quiet --ignore-installed blinker; \
    pip install --quiet quart msgpack aiohttp pyzmq'
RUN git clone --recursive "${MORI_REPO}" /tmp/mori-src && \
    cd /tmp/mori-src && git checkout "${MORI_REF}" && git submodule update --init --recursive && \
    MORI_GPU_ARCHS=${GFX_ARCH} python3 setup.py bdist_wheel --dist-dir=dist && \
    { pip uninstall -y mori || true; } && \
    pip install --no-deps dist/*.whl && \
    echo "MORI_REF=${MORI_REF}@$(git rev-parse HEAD) NIC_BACKEND=${NIC_BACKEND}" >> /app/versions.txt && \
    cd / && rm -rf /tmp/mori-src && \
    python3 -c "import mori; print('mori ok', mori.__file__)"

# -----------------------------------------------------------------------------
# 2. vLLM from source. Same steps as upstream Dockerfile.rocm build_vllm +
#    final_common, collapsed into one stage. setup.py builds vllm-rs itself.
# -----------------------------------------------------------------------------
ARG VLLM_REPO=https://github.com/basemam/vllm.git
ARG VLLM_REF=fff6f12c4ec0c51f80220121ea83d1ce8de16c7e
ARG ROCM_TRITON_KERNELS_COMMIT=669b31acc1dd1b3fd93286afd5db67f65d9f7557
ENV VLLM_TARGET_DEVICE=rocm \
    CARGO_HOME=/root/.cargo \
    RUSTUP_HOME=/root/.rustup \
    PATH=/root/.cargo/bin:${PATH} \
    CARGO_BUILD_JOBS=8 \
    TRITON_KERNELS_SRC_DIR=/opt/rocm-triton-kernels
RUN git clone "${VLLM_REPO}" /app/vllm && \
    cd /app/vllm && git checkout "${VLLM_REF}" && \
    test "$(git rev-parse HEAD)" = "${VLLM_REF}" && \
    grep -Fq "set(TRITON_KERNELS_TAG \"${ROCM_TRITON_KERNELS_COMMIT}\")" cmake/external_projects/triton_kernels.cmake && \
    TOOLCHAIN="$(grep '^channel' rust-toolchain.toml | sed 's/.*= *"\(.*\)"/\1/')" && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none && \
    rustup toolchain install "${TOOLCHAIN}" && rustup default "${TOOLCHAIN}" && \
    mkdir -p /opt/rocm-triton-kernels && \
    curl -fsSL "https://codeload.github.com/ROCm/triton/tar.gz/${ROCM_TRITON_KERNELS_COMMIT}" | \
      tar -xz -C /opt/rocm-triton-kernels --strip-components=4 --no-same-owner \
        "triton-${ROCM_TRITON_KERNELS_COMMIT}/python/triton_kernels/triton_kernels" && \
    test -f /opt/rocm-triton-kernels/__init__.py && \
    uv pip install --system -r requirements/rocm.txt -r requirements/build/rust.txt && \
    { ulimit -n 1048576 || ulimit -n 65536 || true; } && \
    LDFLAGS="-fuse-ld=mold" python3 setup.py bdist_wheel --dist-dir=dist && \
    echo "opentelemetry-api<=1.40.0" > /tmp/otel-constraints.txt && \
    { pip uninstall -y vllm || true; } && \
    uv pip install --system -c /tmp/otel-constraints.txt -r requirements/rocm.txt dist/*.whl && \
    rm -rf /tmp/otel-constraints.txt build dist && \
    echo "VLLM_REF=${VLLM_REF}@$(git rev-parse HEAD) $(git log -1 --format=%cI) ${VLLM_REPO}" >> /app/versions.txt && \
    echo "TRITON_KERNELS_REF=${ROCM_TRITON_KERNELS_COMMIT}" >> /app/versions.txt && \
    echo "RUST_VLLM=$(rustc --version)" >> /app/versions.txt && \
    cd / && python3 -c "import vllm; print('vllm', vllm.__version__, vllm.__file__)" && \
    python3 -c "import vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector" && \
    python3 -c "import vllm.model_executor.layers.fused_moe.prepare_finalize.mori"

# -----------------------------------------------------------------------------
# 3. vllm-router from source -> /usr/local/bin/vllm-router.
#    cargo needs the raised nofile (docker build --ulimit) or it hits EMFILE.
#    The router builds vLLM's rust tokenizer crate (vllm v0.29.0), which uses
#    str::floor_char_boundary (stable in 1.91), so it takes vLLM's own
#    rust-toolchain.toml channel. openssl-sys needs libssl-dev + pkg-config.
# -----------------------------------------------------------------------------
ARG ROUTER_REPO=https://github.com/vllm-project/router.git
ARG ROUTER_REF=37b37a43c577cd7d83bad7b10b8b4650325b4fc7
ARG ROUTER_CRATE=0.1.16
ARG ROUTER_RUST=1.95
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libssl-dev pkg-config && \
    rm -rf /var/lib/apt/lists/* && \
    rustup toolchain install "${ROUTER_RUST}" && \
    git clone --filter=blob:none "${ROUTER_REPO}" /tmp/vllm-router-src && \
    cd /tmp/vllm-router-src && git checkout "${ROUTER_REF}" && \
    test "$(git rev-parse HEAD)" = "${ROUTER_REF}" && \
    grep -q "version = \"${ROUTER_CRATE}\"" Cargo.toml && \
    { ulimit -n 1048576 || ulimit -n 65536 || true; } && \
    cargo +"${ROUTER_RUST}" build --release --locked -j "${CARGO_BUILD_JOBS}" && \
    install -m 755 target/release/vllm-router /usr/local/bin/vllm-router && \
    vllm-router --help 2>&1 | grep -q moriio && \
    echo "VLLM_ROUTER_REF=${ROUTER_REF}@$(git rev-parse HEAD) crate ${ROUTER_CRATE} rustc ${ROUTER_RUST}" >> /app/versions.txt && \
    cd / && rm -rf /tmp/vllm-router-src

# -----------------------------------------------------------------------------
# Runtime env. Upstream final_common values, except HSA_ENABLE_IPC_MODE_LEGACY:
# upstream sets 1 for UCX rocm_ipc; the MoRIIO PD path was validated with 0.
# -----------------------------------------------------------------------------
ENV HSA_ENABLE_IPC_MODE_LEGACY=0 \
    TOKENIZERS_PARALLELISM=false \
    SAFETENSORS_FAST_GPU=1 \
    HIP_FORCE_DEV_KERNARG=1 \
    PYTORCH_NVML_BASED_CUDA_CHECK=1 \
    AITER_JIT_DIR=/opt/vllm_cache/aiter_jit \
    VLLM_CACHE_ROOT=/opt/vllm_cache/vllm \
    TRITON_CACHE_DIR=/opt/vllm_cache/triton \
    COMGR_CACHE_DIR=/opt/vllm_cache/comgr

RUN f="$(python3 -c 'import os,vllm;print(os.path.join(os.path.dirname(vllm.__file__),"model_executor/layers/fused_moe/prepare_finalize/mori.py"))')" && \
    grep -q _num_live_dispatch_rows "$f" && \
    echo "MORI_DISPATCH_TRIM=in-tree" >> /app/versions.txt && \
    rm -rf /root/.mori /tmp/mori_jit_* && mkdir -p /root/.mori && \
    echo "=== Build versions ===" && cat /app/versions.txt

CMD ["/bin/bash"]
