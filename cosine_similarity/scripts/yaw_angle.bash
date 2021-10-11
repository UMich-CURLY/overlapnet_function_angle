cd build && make -j6 && cd .. && \

for i in 03
do
	./build/bin/yaw_angle_versus_function_angle /home/cel/media/data/kitti/sequences/$i/poses.txt /home/cel/media/data/kitti/sequences/$i/calib.txt /home/cel/media/data/kitti/sequences/$i/velodyne/ roll_angle_function_angle_seq_$i.csv

done
