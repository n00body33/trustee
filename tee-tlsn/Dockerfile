# A Go base image is enough to build nitriding reproducibly.
# We use a specific instead of the latest image to ensure reproducibility.
FROM golang:1.22 as builder_nitriding

WORKDIR /

# Clone the repository and build the stand-alone nitriding executable.
RUN git clone https://github.com/brave/nitriding-daemon.git
ARG TARGETARCH
RUN ARCH=${TARGETARCH} make -C nitriding-daemon/ nitriding

# Use the intermediate builder image to add our files.  This is necessary to
# avoid intermediate layers that contain inconsistent file permissions.
COPY  start.sh /bin/
RUN chown root:root  /bin/start.sh
RUN chmod 0755       /bin/start.sh


FROM rust:bookworm AS builder_notary
RUN rustup default 1.79.0

WORKDIR /usr/src/tlsn
COPY . .
RUN pwd && echo "Current path logged"  # Added this line to log the current path

RUN cargo install --path crates/notary/server

FROM ubuntu:latest
WORKDIR /root/app
# Install pkg-config and libssl-dev for async-tungstenite to use (as explained above)
RUN apt-get update && apt-get -y upgrade && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
# Create the app directory explicitly
RUN mkdir -p /app/fixture/tls /app/config

# Copy files
COPY ./crates/notary/server/fixture /app/fixture/

COPY ./crates/notary/server/config/config_prod.yaml /app/config/config.yaml

COPY ./tls/notary.crt /app/fixture/tls/notary.crt
COPY ./tls/notary.key /app/fixture/tls/notary.key

# Check if providers.json exists before copying
COPY ./providers.json /app/providers.json

# Now this should work
RUN ls -laR /app

COPY --from=builder_notary /usr/local/cargo/bin/notary-server /app/

# Label to link this image with the repository in Github Container Registry (https://docs.github.com/en/packages/learn-github-packages/connecting-a-repository-to-a-package#connecting-a-repository-to-a-container-image-using-the-command-line)
LABEL org.opencontainers.image.source=https://github.com/tlsnotary/tlsn
LABEL org.opencontainers.image.description="An implementation of the notary server in Rust."


COPY --from=builder_nitriding /nitriding-daemon/nitriding /bin/start.sh /bin/

ENV RUN_IN_CONTAINER="True"

CMD ["start.sh"]