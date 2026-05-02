# syntax=docker/dockerfile:1
FROM rust:1.85-alpine@sha256:4333721398de61f53ccbe53b0b855bcc4bb49e55828e8f652d7a8ac33dd0c118 AS builder
RUN apk add --no-cache musl-dev pkgconfig openssl-dev
WORKDIR /app

COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release 2>/dev/null || true

COPY src/ src/
RUN cargo build --release

FROM alpine:3.20@sha256:d9e853e87e55526f6b2917df91a2115c36dd7c696a35be12163d44e6e2a4b6bc
RUN apk add --no-cache ca-certificates tzdata
COPY --from=builder /app/target/release/aurelius-gateway /usr/local/bin/

EXPOSE 8080
ENV GATEWAY_HOST=0.0.0.0
ENV GATEWAY_PORT=8080

ENTRYPOINT ["aurelius-gateway"]
