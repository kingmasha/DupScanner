Checking the current CuPy install command so the VS Code steps match the package names in use now.In VS Code, open the terminal with Ctrl+ (Control+ on macOS). Confirm the prompt is the interpreter you want — bottom-right status bar, or run python --version. CuPy wheels need Python 3.10+ and an NVIDIA GPU.Check the driver CUDA version:bash

nvidia-smi

Use the CUDA version in the top-right of that output. Then install the matching wheel. Do not pip install cupy — that package is a source build and will fail or compile for a long time.CUDA 12.x:bash

python -m pip install cupy-cuda12x

CUDA 13.x:bash

python -m pip install cupy-cuda13x

If you only have the NVIDIA driver and no system CUDA Toolkit, add the [ctk] extra so the needed CUDA libraries come from PyPI:bash

python -m pip install "cupy-cuda12x[ctk]"

Swap in cupy-cuda13x if nvidia-smi shows CUDA 13.If the terminal is using a venv, activate it first (VS Code usually does this when the folder interpreter is set). Installing with python -m pip puts CuPy on that same interpreter.Verify:bash

python -c "import cupy as cp; print(cp.__version__); print(cp.cuda.runtime.getDeviceCount())"

A device count of 1 or more means the GPU is visible. If import fails, the usual causes are no NVIDIA driver, a CUDA major version that does not match the wheel, or the package installed into a different Python than the one VS Code is running.

