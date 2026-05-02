# syntax=docker/dockerfile:1
FROM rust:1.88-alpine@sha256:9dfaae478ecd298b6b5a039e1f2cc4fc040fc818a2de9aa78fa714dea036574d AS builder
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
