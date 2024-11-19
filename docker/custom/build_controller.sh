# Build the Model Controller for chatterina
# This controller enables calling models deployed with the vllm image we built
# Aside from the regular endpoints of the fastchat model-controller, this image also has
# /v1/completions and /v1/chat/completitions, which uses the openai client to query the vllm model

docker build --no-cache -f Dockerfile --build-arg GH_TOKEN=$GH_TOKEN --build-arg GH_TOKEN_PUB=$GH_TOKEN_PUB -t us.icr.io/chatterina/fsctrl:cuda12.5_openai_dev .
