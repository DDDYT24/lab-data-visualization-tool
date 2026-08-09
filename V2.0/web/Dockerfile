FROM node:24.17.0-bookworm-slim AS dependencies

WORKDIR /app/web
COPY web/package.json web/package-lock.json ./
RUN npm ci

FROM node:24.17.0-bookworm-slim AS builder

ENV NEXT_TELEMETRY_DISABLED=1
WORKDIR /app
COPY --from=dependencies /app/web/node_modules ./web/node_modules
COPY contracts ./contracts
COPY assets ./assets
COPY web ./web
WORKDIR /app/web
RUN npm run build

FROM node:24.17.0-bookworm-slim AS runner

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    HOSTNAME=0.0.0.0 \
    PORT=3000

WORKDIR /app
COPY --from=builder --chown=node:node /app/web/.next/standalone ./
COPY --from=builder --chown=node:node /app/web/.next/static ./web/.next/static

USER node

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["node", "-e", "fetch('http://127.0.0.1:3000/').then(r=>{if(!r.ok)process.exit(1)}).catch(()=>process.exit(1))"]

CMD ["node", "web/server.js"]
