FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspaces/roughpy-jax

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

COPY pyproject.toml /workspaces/roughpy-jax/

# Create the project environment during image build, but do not compile the
# heavy native RoughPy dependency here. We'll build it interactively if needed.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra dev \
    --no-install-project \
    --no-install-package roughpy-jax \
    --no-install-package roughpy

# Keep native CMake builds conservative inside the devcontainer.
ENV CMAKE_BUILD_PARALLEL_LEVEL=1
