## overlap_gpu
This docker file sets up the environment for overlap_gpu. It aims at installing all non-conflicting related softwares, and encourages download-and-run. We use Nvidia-Docker 2

### How to build the docker image from `Dockerfile`?
`docker pull nvcr.io/nvidia/tensorflow:21.02-tf2-py3`
To build the docker image, run `docker build --tag umcurly/overlap_nvidia .`

### How to run this docker container?
`bash run_docker.bash [container_name]`. Change the home directly, disk volumn mapping in this bash file correspondingly.

### After the docker container is running 
`docker exec -it [container_name] /bin/bash` to use bash as user
`docker exec -u root -it [container_name] /bin/bash` to use bash as root
