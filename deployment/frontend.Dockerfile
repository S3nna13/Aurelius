# syntax=docker/dockerfile:1
FROM node:26-alpine@sha256:e71ac5e964b9201072425d59d2e876359efa25dc96bb1768cb73295728d6e4ea AS builder
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
RUN chown -R nginx:nginx /usr/share/nginx/html
USER nginx

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
