cd build && make -j6 && cd .. && \

for i in 10
do
	for frame_id in 691
	do

	./build/bin/generate_function_angle /home/cel/media/data/kitti/sequences/$i/poses.txt /home/cel/media/data/kitti/sequences/$i/calib.txt /home/cel/media/data/kitti/sequences/$i/velodyne/ function_angle_seq_$i-$frame_id.csv $frame_id
	done

done
