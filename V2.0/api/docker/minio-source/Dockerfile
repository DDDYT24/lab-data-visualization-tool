# Integration-test services only: not included in the Windows Local installer.
# Official legacy images are unavailable; build pinned upstream source instead.
FROM golang:1.24.2-bookworm AS minio-builder
RUN git clone --depth 1 --branch RELEASE.2025-04-22T22-12-26Z https://github.com/minio/minio.git /src/minio
WORKDIR /src/minio
RUN CGO_ENABLED=0 go build -trimpath -o /out/minio .

FROM golang:1.24.2-bookworm AS mc-builder
RUN git clone --depth 1 --branch RELEASE.2025-04-16T18-13-26Z https://github.com/minio/mc.git /src/mc
WORKDIR /src/mc
RUN CGO_ENABLED=0 go build -trimpath -o /out/mc .

FROM debian:bookworm-slim AS base
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*

FROM base AS minio
COPY --from=minio-builder /out/minio /usr/local/bin/minio
COPY --from=minio-builder /src/minio/LICENSE /usr/share/doc/minio/LICENSE
ENTRYPOINT ["minio"]

FROM base AS mc
COPY --from=mc-builder /out/mc /usr/local/bin/mc
COPY --from=mc-builder /src/mc/LICENSE /usr/share/doc/mc/LICENSE
ENTRYPOINT ["mc"]
