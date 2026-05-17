# syntax=docker/dockerfile:1
FROM node:22-alpine@sha256:8ea2348b068a9544dae7317b4f3aafcdc032df1647bb7d768a05a5cad1a7683f AS builder
WORKDIR /app

COPY package.json package-lock.json ./
COPY frontend/package.json frontend/package.json
COPY middle/package.json middle/package.json
COPY server/package.json server/package.json
COPY crates/data-engine/package.json crates/data-engine/package.json
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
COPY frontend/ frontend/
RUN npm run build -w frontend

FROM nginx:stable-alpine@sha256:6525b050aa05151ca19ec7090851bc8c12006cffdae5187f3d28023402f44cfa
COPY --from=builder /app/frontend/dist /usr/share/nginx/html
COPY deployment/nginx.conf /etc/nginx/conf.d/default.conf
RUN chown -R nginx:nginx /usr/share/nginx/html
USER nginx

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
