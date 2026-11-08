//! Pure-Rust forward pass of the LeNet-style digit classifier in
//! `recognise/digitsp.keras`.
//!
//! The weights are exported verbatim by `tools/export_weights.py`; the network
//! is not retrained or altered. Architecture (input 32x32x1, raw 0..255 pixels):
//!
//! conv 5x5x6 tanh → avgpool 2 → sigmoid → conv 5x5x16 tanh → avgpool 2 →
//! sigmoid → conv 5x5x120 tanh → flatten → dense 84 tanh → dense 11 softmax
//!
//! Classes 0..=9 are digits, class 10 is a blank (unlit) digit.

use anyhow::{bail, ensure, Result};

pub const INPUT_SIZE: usize = 32;
pub const CLASSES: usize = 11;
pub const BLANK_CLASS: usize = 10;

const MAGIC: &[u8; 8] = b"WMLENET1";
const WEIGHTS: &[u8] = include_bytes!("../assets/digits_weights.bin");

/// Channels-last feature map.
struct Tensor {
    h: usize,
    w: usize,
    c: usize,
    data: Vec<f32>,
}

impl Tensor {
    fn map(mut self, f: fn(f32) -> f32) -> Self {
        self.data.iter_mut().for_each(|x| *x = f(*x));
        self
    }
}

struct Conv {
    k: usize,
    cin: usize,
    cout: usize,
    /// HWIO layout, as stored by Keras.
    kernel: Vec<f32>,
    bias: Vec<f32>,
}

impl Conv {
    /// 'valid' padding, stride 1 (Keras Conv2D is a cross-correlation).
    fn forward(&self, x: &Tensor) -> Tensor {
        debug_assert_eq!(x.c, self.cin);
        let (oh, ow) = (x.h - self.k + 1, x.w - self.k + 1);
        let mut out = Vec::with_capacity(oh * ow * self.cout);
        for oy in 0..oh {
            for ox in 0..ow {
                let mut acc = self.bias.clone();
                for ky in 0..self.k {
                    for kx in 0..self.k {
                        let px = ((oy + ky) * x.w + ox + kx) * x.c;
                        for i in 0..self.cin {
                            let v = x.data[px + i];
                            let row = ((ky * self.k + kx) * self.cin + i) * self.cout;
                            let weights = &self.kernel[row..row + self.cout];
                            acc.iter_mut().zip(weights).for_each(|(a, w)| *a += v * w);
                        }
                    }
                }
                out.extend(acc);
            }
        }
        Tensor {
            h: oh,
            w: ow,
            c: self.cout,
            data: out,
        }
    }
}

struct Dense {
    nin: usize,
    nout: usize,
    /// in x out layout, as stored by Keras.
    kernel: Vec<f32>,
    bias: Vec<f32>,
}

impl Dense {
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.nin);
        let mut acc = self.bias.clone();
        for (i, v) in x.iter().enumerate() {
            let row = &self.kernel[i * self.nout..(i + 1) * self.nout];
            acc.iter_mut().zip(row).for_each(|(a, w)| *a += v * w);
        }
        acc
    }
}

/// 2x2 average pooling, stride 2, 'valid' padding.
fn avg_pool(x: &Tensor) -> Tensor {
    let (oh, ow) = (x.h / 2, x.w / 2);
    let mut out = vec![0f32; oh * ow * x.c];
    for oy in 0..oh {
        for ox in 0..ow {
            for c in 0..x.c {
                let at = |y: usize, xx: usize| x.data[(y * x.w + xx) * x.c + c];
                let (y, xx) = (oy * 2, ox * 2);
                out[(oy * ow + ox) * x.c + c] =
                    (at(y, xx) + at(y, xx + 1) + at(y + 1, xx) + at(y + 1, xx + 1)) / 4.0;
            }
        }
    }
    Tensor {
        h: oh,
        w: ow,
        c: x.c,
        data: out,
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softmax(x: &[f32]) -> [f32; CLASSES] {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut out = [0f32; CLASSES];
    out.iter_mut()
        .zip(x)
        .for_each(|(o, v)| *o = (v - max).exp());
    let sum: f32 = out.iter().sum();
    out.iter_mut().for_each(|o| *o /= sum);
    out
}

pub struct DigitModel {
    conv1: Conv,
    conv2: Conv,
    conv3: Conv,
    dense1: Dense,
    dense2: Dense,
}

impl DigitModel {
    /// Loads the weights embedded in the binary.
    pub fn load() -> Result<Self> {
        Self::from_bytes(WEIGHTS)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let Some(payload) = bytes.strip_prefix(MAGIC.as_slice()) else {
            bail!("digit model weights: bad magic header");
        };
        let (chunks, rest) = payload.as_chunks::<4>();
        ensure!(rest.is_empty(), "digit model weights: truncated file");
        let mut floats = chunks.iter().map(|&b| f32::from_le_bytes(b));
        let mut take = |n: usize| -> Result<Vec<f32>> {
            let v: Vec<f32> = floats.by_ref().take(n).collect();
            ensure!(v.len() == n, "digit model weights: file too short");
            Ok(v)
        };
        let mut conv = |k: usize, cin: usize, cout: usize| -> Result<Conv> {
            Ok(Conv {
                k,
                cin,
                cout,
                kernel: take(k * k * cin * cout)?,
                bias: take(cout)?,
            })
        };
        let conv1 = conv(5, 1, 6)?;
        let conv2 = conv(5, 6, 16)?;
        let conv3 = conv(5, 16, 120)?;
        let mut dense = |nin: usize, nout: usize| -> Result<Dense> {
            Ok(Dense {
                nin,
                nout,
                kernel: take(nin * nout)?,
                bias: take(nout)?,
            })
        };
        let dense1 = dense(120, 84)?;
        let dense2 = dense(84, CLASSES)?;
        ensure!(
            floats.next().is_none(),
            "digit model weights: trailing data"
        );
        Ok(Self {
            conv1,
            conv2,
            conv3,
            dense1,
            dense2,
        })
    }

    /// Class probabilities for a 32x32 image of raw (0..255) grey levels,
    /// in row-major order.
    pub fn predict(&self, input: &[f32]) -> [f32; CLASSES] {
        assert_eq!(
            input.len(),
            INPUT_SIZE * INPUT_SIZE,
            "digit model input must be 32x32"
        );
        let x = Tensor {
            h: INPUT_SIZE,
            w: INPUT_SIZE,
            c: 1,
            data: input.to_vec(),
        };
        let x = self.conv1.forward(&x).map(f32::tanh);
        let x = avg_pool(&x).map(sigmoid);
        let x = self.conv2.forward(&x).map(f32::tanh);
        let x = avg_pool(&x).map(sigmoid);
        let x = self.conv3.forward(&x).map(f32::tanh);
        let x: Vec<f32> = self
            .dense1
            .forward(&x.data)
            .into_iter()
            .map(f32::tanh)
            .collect();
        softmax(&self.dense2.forward(&x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_weights_load() {
        DigitModel::load().unwrap();
    }

    #[test]
    fn rejects_corrupt_weights() {
        assert!(DigitModel::from_bytes(b"nope").is_err());
        assert!(DigitModel::from_bytes(&WEIGHTS[..WEIGHTS.len() - 4]).is_err());
    }

    /// Reference outputs computed with Keras on the original model; see
    /// `tests/fixtures/README.md`.
    #[test]
    fn matches_keras_reference() {
        let reference: Vec<Vec<f32>> =
            serde_json::from_str(include_str!("../tests/fixtures/keras_reference.json")).unwrap();
        let model = DigitModel::load().unwrap();
        for (seed, expected) in reference.iter().enumerate() {
            let input = reference_input(seed as u32);
            let got = model.predict(&input);
            for (g, e) in got.iter().zip(expected) {
                assert!(
                    (g - e).abs() < 1e-4,
                    "seed {seed}: got {got:?}, expected {expected:?}"
                );
            }
        }
    }

    /// Deterministic test pattern, mirrored in `tools/keras_reference.py`.
    pub fn reference_input(seed: u32) -> Vec<f32> {
        (0..(INPUT_SIZE * INPUT_SIZE) as u32)
            .map(|i| ((i * (37 + 2 * seed) + seed * 101) % 256) as f32)
            .collect()
    }
}
