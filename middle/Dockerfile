# syntax=docker/dockerfile:1
FROM node:22-alpine@sha256:8ea2348b068a9544dae7317b4f3aafcdc032df1647bb7d768a05a5cad1a7683f AS builder
WORKDIR /app

RUN apk add --no-cache build-base cargo rust

COPY package.json package-lock.json ./
COPY middle/package.json middle/package.json
COPY frontend/package.json frontend/package.json
COPY server/package.json server/package.json
COPY crates/data-engine/package.json crates/data-engine/package.json
COPY crates/data-engine/index.js crates/data-engine/index.js
COPY crates/data-engine/index.d.ts crates/data-engine/index.d.ts
COPY crates/json-validator/package.json crates/json-validator/package.json
COPY crates/prompt-templates/package.json crates/prompt-templates/package.json
COPY crates/redis-client/package.json crates/redis-client/package.json
COPY crates/search-index/package.json crates/search-index/package.json
COPY crates/session-manager/package.json crates/session-manager/package.json
COPY crates/text-processor/package.json crates/text-processor/package.json
COPY crates/token-counter/package.json crates/token-counter/package.json
COPY crates/uuid-gen/package.json crates/uuid-gen/package.json
COPY crates/vector-similarity/package.json crates/vector-similarity/package.json
RUN npm ci

COPY crates/data-engine/Cargo.toml crates/data-engine/Cargo.lock crates/data-engine/build.rs crates/data-engine/
COPY crates/data-engine/src crates/data-engine/src
RUN npm run build -w aurelius-data-engine

COPY middle/ middle/
RUN npm run build -w middle

FROM node:22-alpine@sha256:8ea2348b068a9544dae7317b4f3aafcdc032df1647bb7d768a05a5cad1a7683f
WORKDIR /app

RUN apk add --no-cache libc6-compat

COPY --from=builder /app/node_modules ./node_modules
RUN mkdir -p ./crates/data-engine
COPY --from=builder /app/crates/data-engine/package.json ./crates/data-engine/package.json
COPY --from=builder /app/crates/data-engine/index.js ./crates/data-engine/index.js
COPY --from=builder /app/crates/data-engine/index.d.ts ./crates/data-engine/index.d.ts
COPY --from=builder /app/crates/data-engine/aurelius-data-engine.*.node ./crates/data-engine/
COPY --from=builder /app/middle/dist ./dist
COPY --from=builder /app/middle/package.json ./

EXPOSE 3001

ENV NODE_ENV=production
CMD ["node", "dist/index.js"]
