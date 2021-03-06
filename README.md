# OverlapNet with Function Angle

This repository is based on [PRBonn/OverlapNet](https://github.com/PRBonn/OverlapNet). OverlapNet is modified Siamese Network that predicts the overlap and relative yaw angle of a pair of range images generated by 3D LiDAR scans. You may see details from their [paper](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/chen2020rss.pdf).

This repository use the cosine similarity of two functions that represent two point clouds instead of the overlap value proposed in OverlapNet. Using a similar architecture of OverlapNet to learn and estimate the cosine similarity and relative yaw angle between two point clouds.

## Installation
* See instructions in the [docker folder](docker/) to build the docker for the environment.
* This repository is tested on Ubuntu 20.04 with Tensorflow-2.4.1

## Cosine Similarity
* The cosine similarity of two point clouds are calculated using the code in the [cosine_similarity](cosine_similarity/) folder which is a simplified code of [UMich-CURLY/unified_cvo](https://github.com/UMich-CURLY/unified_cvo) for generating the cosine similarities over two point clouds.

## Run OverlapNet
* To generate groundtruth, training and testing data, run `python3 demo/gen_continuous_groundtruth.py config/demo.yml` 
* To train the network, run `python src/two_heads/training.py config/network.yml`
* To test the network, run `python src/two_heads/testing.py config/test.yml`
