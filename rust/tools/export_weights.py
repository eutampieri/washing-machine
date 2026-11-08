#!/usr/bin/env python3
"""Export the weights of recognise/digitsp.keras to a flat binary file.

The model itself is not modified: every tensor is copied verbatim, in Keras'
native layout (conv kernels HWIO, dense kernels in x out), as little-endian
float32, after an 8-byte magic header. Only h5py and numpy are required.

Usage: python3 export_weights.py ../../recognise/digitsp.keras ../assets/digits_weights.bin
"""
import io
import sys
import zipfile

import h5py
import numpy as np

MAGIC = b"WMLENET1"
# (layer name, kernel shape, bias shape), in forward-pass order.
LAYERS = [
    ("conv2d", (5, 5, 1, 6), (6,)),
    ("conv2d_1", (5, 5, 6, 16), (16,)),
    ("conv2d_2", (5, 5, 16, 120), (120,)),
    ("dense", (120, 84), (84,)),
    ("dense_1", (84, 11), (11,)),
]


def main(src: str, dst: str) -> None:
    with zipfile.ZipFile(src) as z:
        weights = h5py.File(io.BytesIO(z.read("model.weights.h5")), "r")
        with open(dst, "wb") as out:
            out.write(MAGIC)
            for name, kernel_shape, bias_shape in LAYERS:
                for var, shape in (("0", kernel_shape), ("1", bias_shape)):
                    data = np.asarray(weights[f"layers/{name}/vars/{var}"], dtype="<f4")
                    if data.shape != shape:
                        raise SystemExit(f"{name}/{var}: expected {shape}, got {data.shape}")
                    out.write(np.ascontiguousarray(data).tobytes())


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
