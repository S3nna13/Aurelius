# syntax=docker/dockerfile:1
FROM node:22-alpine@sha256:8ea2348b068a9544dae7317b4f3aafcdc032df1647bb7d768a05a5cad1a7683f AS builder
WORKDIR /app

COPY package.json package-lock.json ./
COPY frontend/package.json frontend/package.json
COPY middle/package.json middle/package.json
COPY server/package.json server/package.json
COPY crates/data-engine/package.json crates/data-engine/package.json
RUN for d in crates/json-validator crates/prompt-templates crates/redis-client crates/search-index crates/session-manager crates/text-processor crates/token-counter crates/uuid-gen crates/vector-similarity; do mkdir -p "$d" && echo '{"private":true}' > "$d/package.json"; done
RUN npm ci && npm run build -w frontend

FROM nginx:stable-alpine@sha256:f07bea4d6e97d399c7984460011460d98e58972420128ba1cceb3f091fbda25f
COPY --from=builder /app/frontend/dist /usr/share/nginx/html
COPY deployment/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
