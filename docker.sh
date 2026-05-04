# 1. Build the image
docker build --platform linux/amd64 -t ghost-lanes-attack .

# 2. Run the experiment (linking to your Wineskin server)
docker run --platform linux/amd64 --rm -it \
  --memory="2g" \
  --add-host=host.docker.internal:host-gateway \
  -e CARLA_HOST=localhost \
  ghost-lanes-attack /bin/bash