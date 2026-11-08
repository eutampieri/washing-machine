# Build from the repository root (the templates live in recognise/):
#   docker build -f rust/Dockerfile -t washing-machine-monitor .
# Builder and runtime use the same Debian release, so the OpenCV shared
# libraries linked at build time are the ones installed at run time.
FROM docker.io/library/rust:1-trixie AS builder
RUN apt-get update \
 && apt-get install -y --no-install-recommends libopencv-dev clang libclang-dev \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY recognise/display_template.png recognise/template_bar.jpg recognise/
COPY rust/ rust/
WORKDIR /src/rust
RUN cargo build --release --locked

FROM docker.io/library/debian:trixie-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libopencv-core410 libopencv-imgproc410 libopencv-imgcodecs410 libopencv-videoio410 \
      libopencv-features2d410 libopencv-calib3d410 libopencv-flann410 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /src/rust/target/release/washing-machine-monitor /usr/local/bin/
# RTSP over TCP: no lost UDP packets, hence no half-decoded frames.
# Only fatal FFmpeg messages (joining a stream mid-GOP logs harmless decoder
# errors) and only OpenCV errors; failures are reported by the monitor itself.
ENV OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp" \
    OPENCV_FFMPEG_LOGLEVEL=8 \
    OPENCV_LOG_LEVEL=ERROR
USER nobody
ENTRYPOINT ["/usr/local/bin/washing-machine-monitor"]
