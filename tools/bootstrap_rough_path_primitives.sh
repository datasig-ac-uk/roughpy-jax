#!/usr/bin/env bash

set -euo pipefail

RPP_GIT_URL="${RPP_GIT_URL:-https://github.com/datasig-ac-uk/rough-path-primitives.git}"
RPP_GIT_TAG="${RPP_GIT_TAG:-releases/1.0.2}"
RPP_INSTALL_PREFIX="${RPP_INSTALL_PREFIX:-/usr}"

need_cmd() {
    command -v "$1" >/dev/null 2>&1
}

require_cmd() {
    if ! need_cmd "$1"; then
        printf 'Required command not found: %s\n' "$1" >&2
        exit 1
    fi
}

ensure_ninja() {
    if need_cmd ninja; then
        return
    fi

    require_cmd pipx

    local pipx_bin_dir="${PIPX_BIN_DIR:-$HOME/.local/bin}"
    export PATH="${pipx_bin_dir}:${PATH}"

    if ! need_cmd ninja; then
        pipx install ninja
    fi

    if ! need_cmd ninja; then
        printf 'Failed to provision ninja via pipx.\n' >&2
        exit 1
    fi
}

require_cmd git
require_cmd cmake
ensure_ninja

workdir="$(mktemp -d)"
trap 'rm -rf "${workdir}"' EXIT

git clone --depth 1 --branch "${RPP_GIT_TAG}" "${RPP_GIT_URL}" "${workdir}/src"

cmake_args=(
    -S "${workdir}/src"
    -B "${workdir}/build"
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX="${RPP_INSTALL_PREFIX}"
    -DRPP_ENABLE_TESTS=OFF
    -DRPP_ENABLE_BENCHMARKS=OFF
    -DRPP_ADD_STUB_LIBRARY=OFF
)

if need_cmd ninja; then
    cmake_args+=(-G Ninja)
fi

cmake "${cmake_args[@]}"
cmake --build "${workdir}/build" --parallel
cmake --install "${workdir}/build"
