FROM rust:latest

WORKDIR /app

RUN rustup component add rustfmt --toolchain 1.76.0-aarch64-unknown-linux-gnu
RUN rustup component add clippy