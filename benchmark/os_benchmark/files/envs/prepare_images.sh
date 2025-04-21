docker pull mysql
docker pull ubuntu
docker build -f .local/envs/os/dockerfiles/default .local/envs/os/dockerfiles --tag local-os/default
docker build -f .local/envs/os/dockerfiles/packages .local/envs/os/dockerfiles --tag local-os/packages
docker build -f .local/envs/os/dockerfiles/ubuntu .local/envs/os/dockerfiles --tag local-os/ubuntu
