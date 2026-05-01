# Multi-stage build: Rust engine → Node.js gateway

# Stage 1: Build Rust data engine
FROM rust:1.85-slim AS rust-builder
WORKDIR /engine
RUN rustup target add aarch64-unknown-linux-gnu
COPY crates/data-engine/Cargo.toml crates/data-engine/Cargo.lock* ./
RUN mkdir src && echo "fn main() {}" > src/lib.rs
RUN cargo build --release 2>/dev/null || true
COPY crates/data-engine/ .
RUN cargo build --release

# Stage 2: Build frontend
FROM node:22-alpine AS frontend-builder
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

# Stage 3: Build and run gateway
FROM node:22-slim AS gateway
WORKDIR /app

# Install Rust engine native addon dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libc6 && rm -rf /var/lib/apt/lists/*

# Copy Rust engine
COPY --from=rust-builder /engine/target/release/libaurelius_data_engine.so /app/crates/data-engine/
COPY crates/data-engine/index.js crates/data-engine/index.d.ts /app/crates/data-engine/

# Copy server
COPY server/package*.json /app/server/
RUN cd server && npm ci --omit=dev

COPY server/ /app/server/

# Copy frontend build
COPY --from=frontend-builder /frontend/dist /app/frontend/dist

EXPOSE 7870
ENV NODE_ENV=production
ENV AURELIUS_FRONTEND_DIST=/app/frontend/dist

WORKDIR /app/server
CMD ["node", "dist/main.js"]
