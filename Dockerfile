# Base image
FROM rust:latest

# Download and install libtorch
RUN wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip && \
    unzip libtorch-shared-with-deps-latest.zip -d /usr/local && \
    rm libtorch-shared-with-deps-latest.zip

# Set up the libtorch environment
ENV LIBTORCH=/usr/local/libtorch

# Install Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Set up the Rust environment
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy the Cargo.toml and Cargo.lock files
COPY Cargo.toml Cargo.lock ./

# Build the dependencies (including serde with derive feature)
RUN cargo build --release

# Set up workspace
WORKDIR /workspace