FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspaces/roughpy-jax

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/GMT apt-get update && apt-get install -y \
    build-essential \
    git \
    openssh-client \
    gdb \
    curl \
    zip \
    unzip \
    tar \
    autoconf \
    libtool \
    libtool-bin \
    pkg-config \
    wbritish \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .pre-commit-config.yaml /workspaces/roughpy-jax/

RUN uv venv ${VIRTUAL_ENV}

# Resolve the project dependencies into a stable virtual environment that lives
# outside the bind-mounted workspace. Build RoughPy from source now, but defer
# the editable roughpy-jax install until the container has been created.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --active --extra dev \
    --no-install-project \
    --no-binary-package roughpy

# Install the build backend and toolchain into the runtime environment so the
# editable roughpy-jax install can reuse the already-installed RoughPy package.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python ${VIRTUAL_ENV}/bin/python \
    "scikit-build-core[pyproject]>=0.10" \
    "cmake>=3.30" \
    "ninja>=1.11"