container_name=$1

xhost +local:
docker run -it --net=host --gpus all  \
  --user=$(id -u) \
  -e DISPLAY=$DISPLAY \
  -e QT_GRAPHICSSYSTEM=native \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e XAUTHORITY \
  -e USER=$USER \
  --workdir=/home/$USER/ \
  -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  -v "/etc/passwd:/etc/passwd:rw" \
  -e "TERM=xterm-256color" \
  -v "/home/$USER/DockerFolder:/home/$USER/" \
  --device=/dev/dri:/dev/dri \
  --name=${container_name} \
  umcurly/overlap_nvidia:latest
