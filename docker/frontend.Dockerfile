# ApexPulse dashboard: the Next.js frontend.
#
# Three stages: install dependencies from the lockfile, build, then run with only
# the production dependencies the server needs.

FROM node:22-alpine AS deps
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund


FROM node:22-alpine AS build
WORKDIR /app

# NEXT_PUBLIC_* values are inlined into the browser bundle at build time, not
# read at run time, so they are build arguments. They name where the *viewer's
# browser* reaches the API — the host's address, not a container hostname.
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ARG NEXT_PUBLIC_WS_URL=ws://localhost:8000
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL} \
    NEXT_PUBLIC_WS_URL=${NEXT_PUBLIC_WS_URL} \
    NEXT_TELEMETRY_DISABLED=1

COPY --from=deps /app/node_modules ./node_modules
COPY frontend/ ./
RUN npm run build


FROM node:22-alpine AS runtime
WORKDIR /app

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --omit=dev --no-audit --no-fund && npm cache clean --force

# Owned by `node`, because Next writes its cache under .next/cache at run time
# and would fail on a root-owned directory.
COPY --from=build --chown=node:node /app/.next ./.next
COPY --from=build --chown=node:node /app/next.config.mjs ./next.config.mjs

# The base image ships a `node` user; the server has no reason to run as root.
USER node
EXPOSE 3000

CMD ["npx", "next", "start", "-p", "3000"]
