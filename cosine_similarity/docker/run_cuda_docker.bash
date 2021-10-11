#container_name=$1

#xhost +local:
#docker run -it --net=host \ #--gpus all  \            # this enable nvidia driver
#  --user $(id -u):$(id -g) \
#  -e DISPLAY=$DISPLAY \
#  -e QT_GRAPHICSSYSTEM=native \
#  -e NVIDIA_DRIVER_CAPABILITIES=all \
#  -e XAUTHORITY \
#  -e USER=$USER \                                     # use the same username as you
#  --workdir=/home/$USER/ \
#  -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \           
#  -v "/etc/passwd:/etc/passwd:rw" \        
#  -e "TERM=xterm-256color" \
#  -v "/home/$USER/DockerFolder:/home/$USER/" \        # map the "/home/$USER/code/docker_home" outside docker as the docker container's home dir
#  --device=/dev/dri:/dev/dri \
#  --name=${container_name} \
#  umrobotics/cvo:latest



container_name=$1

xhost +local:
docker run -it --net=host --gpus all  -e DISPLAY=${DISPLAY} \
  -e QT_GRAPHICSSYSTEM=native \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e XAUTHORITY \
  -u `id -u`:`id -g` --workdir="/home/$USER/" \
  -v "/tmp/.X11-unix:/tmp/.X11-unix:rw"  -v "/etc/passwd:/etc/passwd:rw"  -e "TERM=xterm-256color" \
  -v "/home/$USER/DockerFolder:/home/$USER/" \        # map the "/home/$USER/code/docker_home" outside docker as the docker container's home dir
  #--device=/dev/dri:/dev/dri \
  --name=${container_name} \
	  umrobotics/cvo:latest
