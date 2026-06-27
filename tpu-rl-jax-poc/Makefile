.PHONY: build build-cli build-agent clean

OUTDIR = bin/x86_64_Linux

build: build-cli build-agent

build-cli:
	CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -ldflags="-s -w" -o $(OUTDIR)/tpu-checkpoint .

build-agent:
	CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -ldflags="-s -w" -o $(OUTDIR)/snapshot-agent ./cmd/snapshot-agent/

clean:
	rm -rf bin/
