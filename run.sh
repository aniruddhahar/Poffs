#!/usr/bin/env bash
# Volumetric Texture Generator — dependency checker/installer + launcher
# (Linux and macOS; on Windows use run.cmd / run.ps1)
#
# Usage:
#   ./run.sh                     # launch GUI
#   ./run.sh gui                 # launch GUI
#   ./run.sh cli --size 64 ...   # launch CLI (args passed through)
#   ./run.sh --gpu ...           # additionally check/install optional GPU deps
#                                  (pyopencl + numpy)
#
# Checks:
#   - a Python >= 3.10 (GUI additionally requires working tkinter)
#   - optional GPU stack: pyopencl + numpy (only with --gpu)
# Installs are attempted automatically where possible (pip --user, venv
# fallback, apt for python3-tk on Debian/Ubuntu, brew python-tk on macOS).
# Portable across bash 3.2 (macOS default) and bash 4/5.

set -euo pipefail

# cd to the real script directory (readlink -f is not available on macOS)
SOURCE="${BASH_SOURCE[0]:-$0}"
while [ -h "$SOURCE" ]; do
    DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
    SOURCE="$(readlink "$SOURCE")"
    [ "${SOURCE#/}" = "$SOURCE" ] && SOURCE="$DIR/$SOURCE"
done
cd -P "$(dirname "$SOURCE")"

MODE="gui"
WANT_GPU=false
ARGS=()

for arg in "$@"; do
    case "$arg" in
        --gpu) WANT_GPU=true ;;
        gui|cli) if [ ${#ARGS[@]} -eq 0 ]; then MODE="$arg"; else ARGS+=("$arg"); fi ;;
        *) ARGS+=("$arg") ;;
    esac
done

info()  { echo "[run.sh] $*"; }
warn()  { echo "[run.sh] WARNING: $*" >&2; }
have()  { command -v "$1" >/dev/null 2>&1; }

py_ok() {  # interpreter exists and is >= 3.10
    "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1
}

py_tk() {  # interpreter has working tkinter (incl. ttk)
    "$1" -c 'import tkinter, tkinter.ttk' >/dev/null 2>&1
}

# --- 1) Pick interpreter -----------------------------------------------------
PY=""
CANDIDATES=(python3 python3.14 python3.13 python3.12 python3.11 python3.10)

if [ "$MODE" = "gui" ]; then
    for c in "${CANDIDATES[@]}"; do
        if have "$c" && py_ok "$c" && py_tk "$c"; then PY="$c"; break; fi
    done
    if [ -z "$PY" ]; then
        # No interpreter with tkinter — try to install it
        BASE=""
        for c in "${CANDIDATES[@]}"; do
            if have "$c" && py_ok "$c"; then BASE="$c"; break; fi
        done
        if [ -n "$BASE" ] && have apt-get; then
            info "tkinter missing for $BASE — installing via apt (sudo may prompt)"
            sudo apt-get install -y python3-tk || true
            if py_tk "$BASE"; then PY="$BASE"; fi
        elif [ -n "$BASE" ] && have brew; then
            PYVER="$("$BASE" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
            info "tkinter missing for $BASE — trying 'brew install python-tk@$PYVER'"
            brew install "python-tk@$PYVER" >/dev/null 2>&1 \
                || brew install python-tk >/dev/null 2>&1 || true
            if py_tk "$BASE"; then PY="$BASE"; fi
        fi
        if [ -z "$PY" ]; then
            warn "No Python >= 3.10 with tkinter found."
            warn "Install tkinter, e.g.:"
            warn "  Debian/Ubuntu: sudo apt install python3-tk"
            warn "  Fedora:        sudo dnf install python3-tkinter"
            warn "  Arch:          sudo pacman -S tk"
            warn "  macOS:         brew install python-tk@3.12  (or use the"
            warn "                 python.org installer, which bundles tkinter)"
            exit 1
        fi
    fi
else
    for c in "${CANDIDATES[@]}"; do
        if have "$c" && py_ok "$c"; then PY="$c"; break; fi
    done
    if [ -z "$PY" ]; then
        warn "No Python >= 3.10 found. Install Python 3.10 or later."
        exit 1
    fi
fi

info "Interpreter: $PY ($("$PY" --version 2>&1))"

# --- 2) Optional GPU stack (pyopencl + numpy) ---------------------------------
if $WANT_GPU; then
    if "$PY" -c 'import pyopencl, numpy' >/dev/null 2>&1; then
        info "GPU deps already available for $PY"
    else
        info "Installing optional GPU deps (pyopencl, numpy)..."
        GPU_PY="$PY"
        if "$PY" -m pip install --user -q pyopencl numpy 2>/dev/null \
           || "$PY" -m pip install --user --break-system-packages -q pyopencl numpy 2>/dev/null; then
            info "Installed into user site-packages"
        elif "$PY" -m venv --system-site-packages .venv >/dev/null 2>&1 \
             && "$PWD/.venv/bin/pip" install -q pyopencl numpy; then
            GPU_PY="$PWD/.venv/bin/python"
            info "pip --user unavailable; using local .venv (system-site-packages)"
        else
            warn "Could not install pyopencl/numpy — continuing with CPU only."
            GPU_PY=""
        fi
        if [ -n "$GPU_PY" ]; then
            if "$GPU_PY" -c 'import pyopencl as cl; cl.get_platforms()' >/dev/null 2>&1; then
                info "OpenCL platform detected — GPU acceleration available"
            else
                warn "pyopencl installed but no OpenCL platform found — CPU only."
            fi
            PY="$GPU_PY"
        fi
    fi
fi

# --- 3) Display check for GUI (X11/Wayland only; macOS Cocoa & Windows need none)
if [ "$MODE" = "gui" ] && [ "$(uname -s)" = "Linux" ]; then
    if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
        warn "No DISPLAY/WAYLAND_DISPLAY set — GUI will fail to open."
    fi
fi

# --- 4) Launch ----------------------------------------------------------------
if [ "$MODE" = "gui" ]; then
    info "Launching GUI..."
    exec "$PY" generate_volumetric_gui.py
else
    exec "$PY" generate_volumetric.py ${ARGS[@]+"${ARGS[@]}"}
fi
