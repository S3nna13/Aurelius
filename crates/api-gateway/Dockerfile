# syntax=docker/dockerfile:1
FROM rust:1.85-alpine AS builder
RUN apk add --no-cache musl-dev pkgconfig openssl-dev
WORKDIR /app

COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release 2>/dev/null || true

COPY src/ src/
RUN cargo build --release

FROM alpine:3.20
RUN apk add --no-cache ca-certificates tzdata
COPY --from=builder /app/target/release/aurelius-gateway /usr/local/bin/

EXPOSE 8080
ENV GATEWAY_HOST=0.0.0.0
ENV GATEWAY_PORT=8080

ENTRYPOINT ["aurelius-gateway"]
