## CVO
This docker file sets up the environment for CVO. It aims at installing all non-conflicting related softwares, and encourages download-and-run. We use Nvidia-Docker 2


### How to build the docker image from `Dockerfile`?

To build the docker image, run `docker build --tag umrobotics/cvo  . `

If you want to make any changes to this docker image, edit the `Dockerfile`. If any changes happends, remember to update the `LABEL version` inside. 

### How to start a new docker container from this docker image?
`bash run_cuda_docker.bash [container_name]`. Change the home directly, disk volumn mapping in this bash file correspondingly. Note that you only need to do this once. 


### After the docker container is running 
`docker exec -it [container_name] /bin/bash` to use bash as user
`docker exec -u root -it [container_name] /bin/bash` to use bash as root


### when you restart the machine and need to start your old docker container
`docker ps -a` --> Find you old container name 

`docker start [container_name]`  --> start the old container
