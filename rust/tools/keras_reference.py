#!/usr/bin/env python3
"""Generate tests/fixtures/keras_reference.json: Keras outputs of the original
digit model on deterministic test patterns, used to check the Rust port.

Requires tensorflow. Usage: python3 keras_reference.py ../../recognise/digitsp.keras
"""
import json
import sys

import numpy as np
import tensorflow as tf

SEEDS = 5


def reference_input(seed: int) -> np.ndarray:
    # Mirrors `reference_input` in src/model.rs.
    i = np.arange(32 * 32, dtype=np.int64)
    return ((i * (37 + 2 * seed) + seed * 101) % 256).astype(np.float32).reshape(32, 32, 1)


def main(model_path: str) -> None:
    model = tf.keras.models.load_model(model_path)
    batch = np.stack([reference_input(s) for s in range(SEEDS)])
    out = model.predict(batch, verbose=0)
    json.dump([[float(v) for v in row] for row in out], sys.stdout, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
