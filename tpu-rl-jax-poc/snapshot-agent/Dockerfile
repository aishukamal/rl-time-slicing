FROM golang:1.25 AS builder

WORKDIR /src
COPY . .
RUN go mod download
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -ldflags="-s -w" -o /snapshot-agent ./cmd/snapshot-agent/

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=builder /snapshot-agent /snapshot-agent
ENTRYPOINT ["/snapshot-agent"]
CMD ["--port=9001"]
