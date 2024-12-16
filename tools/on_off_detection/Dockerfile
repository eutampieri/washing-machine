FROM rust:latest AS builder
RUN apt update
RUN apt install --no-install-recommends -y libopencv-dev clang libclang-dev
RUN mkdir /build
COPY . /build
WORKDIR /build
RUN cargo build --release
FROM ubuntu:latest
RUN apt update && apt install -y rclone libopencv-imgcodecs-dev libopencv-imgproc-dev libopencv-core-dev && rm -rf /var/cache/apt/archives /var/lib/apt/lists/*
COPY --from=builder /build/target/release/on_off_detection /
ENTRYPOINT /on_off_detection
