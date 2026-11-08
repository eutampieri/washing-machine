# Washing machine monitor

Reads the panel of the washing machine from an RTSP camera and publishes its
state to Home Assistant over MQTT, with automatic discovery.

It is a Rust port of `recognise/recognise.py` (template matching for the
display and the phase bar, the LeNet digit model for the remaining time) and
replaces `tools/on_off_detection`: on/off is derived from the phase bar, found
by template matching, rather than from a fixed region of the frame.

## Home Assistant entities

One device (`DEVICE_NAME`) with:

| Entity | Type | Notes |
|---|---|---|
| Running | `binary_sensor` (`running`) | on when a phase segment is lit |
| Phase | `sensor` (`enum`) | Ammollo/Prelavaggio, Lavaggio, Risciacquo, Stop con acqua, Scarico/Centrifuga, Antipiega/Fine |
| Remaining time | `sensor` (`duration`, min) | unknown when the display cannot be read |
| End time | `sensor` (`timestamp`) | now + remaining time |

The entities become unavailable when the monitor stops (MQTT last will) or
after `FAILURES_BEFORE_UNAVAILABLE` consecutive failed readings (e.g. camera
unreachable). Discovery is re-sent when Home Assistant restarts.

State is published, retained, to `<MQTT_BASE_TOPIC>/state`:

```json
{"running": "ON", "phase": "Risciacquo", "remaining_minutes": 20, "end_time": "2024-12-09T10:15:09Z"}
```

## Configuration

| Variable | Default | |
|---|---|---|
| `RTSP_URL` | — (required) | any URL FFmpeg can open, e.g. `rtsp://user:pass@cam:554/stream1` |
| `CAPTURE_TIMEOUT_SECS` | `10` | open/read timeout |
| `WARMUP_FRAMES` | `5` | frames discarded after connecting |
| `INTERVAL_RUNNING_SECS` | `30` | polling interval while running (and after a failure) |
| `INTERVAL_IDLE_SECS` | `60` | polling interval while off |
| `FAILURES_BEFORE_UNAVAILABLE` | `3` | |
| `MQTT_HOST` / `MQTT_PORT` | `localhost` / `1883` | plain TCP |
| `MQTT_USERNAME` / `MQTT_PASSWORD` | — | |
| `MQTT_CLIENT_ID` | `<DEVICE_ID>_monitor` | |
| `MQTT_BASE_TOPIC` | `<DEVICE_ID>` | |
| `HA_DISCOVERY_PREFIX` | `homeassistant` | |
| `DEVICE_ID` / `DEVICE_NAME` | `washing_machine` / `Washing machine` | |
| `RUST_LOG` | `info` | |
| `OPENCV_FFMPEG_CAPTURE_OPTIONS` | `rtsp_transport;tcp` in the image | FFmpeg options |
| `OPENCV_FFMPEG_LOGLEVEL` / `OPENCV_LOG_LEVEL` | `8` / `ERROR` in the image | library log verbosity |

Frames are converted to grey, rescaled to 1280x720 and registered onto a
reference frame of the dataset (`assets/reference_frame.jpg`): SIFT features of
the machine's front panel are matched and the frame is warped back to the
dataset framing, so the camera does not need to be aimed exactly as when the
dataset was recorded — only the front panel must be in view and in focus.
A frame where the panel cannot be found counts as a failed reading.

## Running

```sh
docker build -f rust/Dockerfile -t washing-machine-monitor .   # from the repository root
docker run -d --restart unless-stopped \
  -e RTSP_URL=rtsp://user:pass@camera/stream -e MQTT_HOST=homeassistant.local \
  -e MQTT_USERNAME=... -e MQTT_PASSWORD=... washing-machine-monitor
```

Offline tools:

```sh
washing-machine-monitor recognise frame.jpg …   # one JSON reading per image
washing-machine-monitor evaluate dataset/       # accuracy on files named like 1734194469R20.jpg
```

To see what the monitor looks at, `tools/roi_demo.py` (Python, needs
`opencv-python` and `numpy`) draws the regions of interest on a frame or a
live stream: the panel used for alignment, the display and its digit cells,
the phase bar with its six bands and the lit one.

```sh
python3 tools/roi_demo.py 'rtsp://user:pass@camera/stream' -o roi.png
```

## Known limitations

* The digit model is fragile outside its training conditions: daylight
  (colour) frames and re-aligned frames from a moved camera can misread
  digits (e.g. 15 → 16, 20 → 28). The test `reads_frames_from_a_moved_camera`
  is ignored for this reason. A 7-segment decoder or a fine-tuned model would fix it.
* The remaining time is only reported while a phase LED is lit.

## Development

Needs OpenCV ≥ 4.5 development files and libclang (`apt install libopencv-dev clang libclang-dev`).

```sh
cargo test
```

The digit model weights in `assets/digits_weights.bin` are an exact copy of
those in `recognise/digitsp.keras`, regenerated with
`python3 tools/export_weights.py ../recognise/digitsp.keras assets/digits_weights.bin`
(needs `h5py`, `numpy`). `tests/fixtures/keras_reference.json` holds Keras'
outputs on fixed inputs, checked against the Rust forward pass; regenerate it
with `tools/keras_reference.py` (needs `tensorflow`).

### Fixes compared with the Python prototype

* digits are padded/resized as in training (to 53 px wide, then 32x32), not padded to 64x64;
* the per-digit limits (hours ≤ 2, tens ≤ 5) are actually applied — `predictions[m+1:9] = 0` sliced the batch axis and changed nothing;
* the "blank digit" class is kept as blank instead of being read as 0; an unreadable display gives an unknown time instead of a wrong one;
* a uniform phase bar no longer divides by zero, and every region is bounds-checked;
* the on/off check no longer uses a fixed region with a threshold that any single bright pixel passed;
* frames are registered onto the dataset framing, so a moved camera still works;
* a phase segment must also be brighter than the rest of the bar (the original
  test only checked that the dark stripe vanished, which a thin or blurred
  stripe also does — the cause of false "running" readings);
* no panics on bad frames, broker or camera outages: errors are logged and retried;
* the evaluation counts failures as errors and computes every metric against the labels.
