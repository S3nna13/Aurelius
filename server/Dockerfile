# Multi-stage build: Rust engine → Node.js gateway

# Stage 1: Build Rust data engine
FROM rust:1.88-slim@sha256:38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89 AS rust-builder
WORKDIR /engine
RUN rustup target add aarch64-unknown-linux-gnu
COPY crates/data-engine/Cargo.toml crates/data-engine/Cargo.lock* ./
RUN mkdir src && echo "fn main() {}" > src/lib.rs
RUN cargo build --release 2>/dev/null || true
COPY crates/data-engine/ .
RUN cargo build --release

# Stage 2: Build frontend
FROM node:22-alpine@sha256:8ea2348b068a9544dae7317b4f3aafcdc032df1647bb7d768a05a5cad1a7683f AS frontend-builder
WORKDIR /app
COPY package.json package-lock.json ./
COPY frontend/package.json frontend/package.json
COPY middle/package.json middle/package.json
COPY server/package.json server/package.json
RUN mkdir -p crates/data-engine crates/json-validator crates/prompt-templates crates/redis-client crates/search-index crates/session-manager crates/text-processor crates/token-counter crates/uuid-gen crates/vector-similarity && \
  touch crates/*/package.json
RUN npm ci && npm run build -w frontend

# Stage 3: Build and run gateway
FROM node:22-slim@sha256:d415caac2f1f77b98caaf9415c5f807e14bc8d7bdea62561ea2fef4fbd08a73c AS gateway
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libc6 && rm -rf /var/lib/apt/lists/*

COPY --from=rust-builder /engine/target/release/libaurelius_data_engine.so /app/crates/data-engine/
COPY crates/data-engine/index.js crates/data-engine/index.d.ts /app/crates/data-engine/

COPY package.json package-lock.json ./
COPY server/package.json server/package.json
COPY frontend/package.json frontend/package.json
RUN mkdir -p crates/data-engine crates/json-validator crates/prompt-templates crates/redis-client crates/search-index crates/session-manager crates/text-processor crates/token-counter crates/uuid-gen crates/vector-similarity && \
  touch crates/*/package.json
RUN npm ci --omit=dev

COPY server/ /app/server/
COPY --from=frontend-builder /app/node_modules/frontend/dist /app/frontend/dist

EXPOSE 7870
ENV NODE_ENV=production
ENV AURELIUS_FRONTEND_DIST=/app/frontend/dist
ENV AURELIUS_AUTH=true

RUN useradd -r -s /bin/false aurelius
USER aurelius

WORKDIR /app/server
CMD ["node", "dist/main.js"]
