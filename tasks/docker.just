# run docker tasks
[group('docker')]
docker: docker-build docker-run

# build docker image
[group('docker')]
docker-build tag="latest": package-build
    docker build --tag={{REPOSITORY}}:{{tag}} -f docker/Dockerfile.python .

# start docker compose
[group('docker')]
docker-compose-service service:
    docker compose up --build {{service}}

# run latest docker image
[group('docker')]
docker-run tag="latest":
    docker run --rm {{REPOSITORY}}:{{tag}}
