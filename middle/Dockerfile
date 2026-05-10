# syntax=docker/dockerfile:1
FROM node@sha256:8ea2348b068a9544dae7317b4f3aafcdc032df1647bb7d768a05a5cad1a7683f AS builder
WORKDIR /app

COPY middle/package.json middle/package-lock.json ./
RUN npm ci

COPY middle/ .
RUN npm run build

FROM node@sha256:8ea2348b068a9544dae7317b4f3aafcdc032df1647bb7d768a05a5cad1a7683f
WORKDIR /app

RUN apk add --no-cache libc6-compat

COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/package.json ./

COPY crates/data-engine/aurelius-data-engine.*.node /app/node_modules/aurelius-data-engine/

EXPOSE 3001

ENV NODE_ENV=production

RUN addgroup -S aurelius && adduser -S -G aurelius aurelius
USER aurelius

CMD ["node", "dist/index.js"]
