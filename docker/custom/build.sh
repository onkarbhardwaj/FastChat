# docker build -f Dockerfile --build-arg GH_TOKEN=$GH_TOKEN --build-arg GH_TOKEN_PUB=$GH_TOKEN_PUB -t us.icr.io/chatterina/vllm:cuda12.5 .


# Base image for testing
docker build --no-cache -f Dockerfile --build-arg GH_TOKEN=$GH_TOKEN --build-arg GH_TOKEN_PUB=$GH_TOKEN_PUB -t us.icr.io/chatterina/vllm:cuda12.5_controller .
