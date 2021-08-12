import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/utils'))
from utils import *

prediction_function_angle = np.empty((0,4))
test_result_filenames = ["/home/cel/DockerFolder/code/overlapnet_function_angle/result/seq00_test_function_angle_ell_1_4000k/validation_results.npz",
                        "/home/cel/DockerFolder/code/overlapnet_function_angle/result/seq00_test_function_angle_ell_1_4000k_end/validation_results.npz",
                        ]
# test_result_filenames = ["/home/cel/DockerFolder/code/overlapnet_function_angle/result/seq00_test_all/seq00_test_function_angle_scratch_900k/validation_results.npz",
#                         "/home/cel/DockerFolder/code/overlapnet_function_angle/result/seq00_test_all/seq00_test_function_angle_scratch_900k_1800k/validation_results.npz",
#                         "/home/cel/DockerFolder/code/overlapnet_function_angle/result/seq00_test_all/seq00_test_function_angle_scratch_1800k_2700k/validation_results.npz",
#                         "/home/cel/DockerFolder/code/overlapnet_function_angle/result/seq00_test_all/seq00_test_function_angle_scratch_2700k_3600k/validation_results.npz",
#                         "/home/cel/DockerFolder/code/overlapnet_function_angle/result/seq00_test_all/seq00_test_function_angle_scratch_3600k_4500k/validation_results.npz",
#                         "/home/cel/DockerFolder/code/overlapnet_function_angle/result/seq00_test_all/seq00_test_function_angle_scratch_4500k_5500k/validation_results.npz",
#                         "/home/cel/DockerFolder/code/overlapnet_function_angle/result/seq00_test_all/seq00_test_function_angle_scratch_5500k_6500k/validation_results.npz",
#                         "/home/cel/DockerFolder/code/overlapnet_function_angle/result/seq00_test_all/seq00_test_function_angle_scratch_6500k_7500k/validation_results.npz",
#                         "/home/cel/DockerFolder/code/overlapnet_function_angle/result/seq00_test_all/seq00_test_function_angle_scratch_7500k_8500k/validation_results.npz",
#                         "/home/cel/DockerFolder/code/overlapnet_function_angle/result/seq00_test_all/seq00_test_function_angle_scratch_8500k_end/validation_results.npz"                           
#                         ]
for test_result_filename in test_result_filenames:
    with np.load(test_result_filename) as data:
        prediction_function_angle = np.vstack((prediction_function_angle, data['arr_0.npy']))

prediction_overlap = np.empty((0,4))
test_result_filenames = ["/home/cel/DockerFolder/code/overlapnet_function_angle/result/seq00_test_all/seq00_test_overlap_2000k/validation_results.npz",
                        "/home/cel/DockerFolder/code/overlapnet_function_angle/result/seq00_test_all/seq00_test_overlap_2000k_4000k/validation_results.npz",
                        "/home/cel/DockerFolder/code/overlapnet_function_angle/result/seq00_test_all/seq00_test_overlap_4000k_6000k/validation_results.npz",
                        "/home/cel/DockerFolder/code/overlapnet_function_angle/result/seq00_test_all/seq00_test_overlap_6000k_8000k/validation_results.npz",
                        "/home/cel/DockerFolder/code/overlapnet_function_angle/result/seq00_test_all/seq00_test_overlap_8000k_end/validation_results.npz"           
                        ]
for test_result_filename in test_result_filenames:
    with np.load(test_result_filename) as data:
        prediction_overlap = np.vstack((prediction_overlap, data['arr_0.npy']))

overlap_test_imgf1 = prediction_overlap[:,0]
overlap_test_imgf2 = prediction_overlap[:,1]
model_outputs_overlap = prediction_overlap[:,2]
model_outputs_yaw_orientation_o = prediction_overlap[:,3]

function_angle_test_imgf1 = prediction_function_angle[:,0]
function_angle_test_imgf2 = prediction_function_angle[:,1]
model_outputs_function_angle = prediction_function_angle[:,2]
model_outputs_yaw_orientation_fa = prediction_function_angle[:,3]


# pointnetvlad
kitti_00_global_features = np.load("/home/cel/code/pointnetvlad_kxhit/pretrained_results/feature_database/fov100/00_PV_ref.npy")

# get ground truth place recognition
f = open('kitti_results/00_triplets/positive_sequence_D-4_T-0.json',)
positive_sequence = json.load(f)
positive_sequence_00 = positive_sequence["00"]
f.close()


# poses_file = "/home/cel/DockerFolder/data/kitti/sequences/00/poses.txt"
# calib_file = "/home/cel/DockerFolder/data/kitti/sequences/00/calib.txt"
# # load calibrations
# T_cam_velo = load_calib(calib_file)
# T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
# T_velo_cam = np.linalg.inv(T_cam_velo)

# # load poses
# poses = load_poses(poses_file)
# pose0_inv = np.linalg.inv(poses[0])

# # for KITTI dataset, we need to convert the provided poses 
# # from the camera coordinate system into the LiDAR coordinate system  
# poses_new = []
# for pose in poses:
#     poses_new.append(T_velo_cam.dot(pose0_inv).dot(pose).dot(T_cam_velo))
# poses = np.array(poses_new)

# # check the indexes are the same
# if np.array_equal(overlap_test_imgf1, function_angle_test_imgf1) and \
#     np.array_equal(overlap_test_imgf2, function_angle_test_imgf2):
#     print('correct')
# else:
#     print('not correct!')

# y_real_true = []
# index_mask_not_near = []
# # go through all pairs, select mask for ignoring near scaans, select ground truth for pose closer than 4 meters
# for index in range(len(overlap_test_imgf1)):
#     idx1 = int(overlap_test_imgf1[index])
#     idx2 = int(overlap_test_imgf2[index])
#     # print(idx1, int(function_angle_test_imgf1[index]), idx2, int(function_angle_test_imgf2[index]))
#     if idx1 - idx2 > 50:
#         index_mask_not_near.append(index)
#         pose1 = poses[idx1]
#         pose2 = poses[idx2]
#         diff = np.linalg.norm(pose1[:3,3] - pose2[:3,3])
#         if diff < 0.3:
#             y_real_true.append(1)
#         else:
#             y_real_true.append(0)

print('ground truth loop closure:', len(y_real_true))
print('index_mask_not_near:', len(index_mask_not_near))

# mask results, only check not near frames
overlap_test_imgf1 = overlap_test_imgf1[index_mask_not_near]
overlap_test_imgf2 = overlap_test_imgf2[index_mask_not_near]
model_outputs_overlap = model_outputs_overlap[index_mask_not_near]
model_outputs_yaw_orientation_o = model_outputs_yaw_orientation_o[index_mask_not_near]

function_angle_test_imgf1 = function_angle_test_imgf1[index_mask_not_near]
function_angle_test_imgf2 = function_angle_test_imgf2[index_mask_not_near]
model_outputs_function_angle = model_outputs_function_angle[index_mask_not_near]
model_outputs_yaw_orientation_fa = model_outputs_yaw_orientation_fa[index_mask_not_near]


# initialize for plotting
roc_x_FPR_fa = []
roc_y_TPR_fa = []
roc_x_FPR_fa2 = []
roc_y_TPR_fa2 = []
roc_x_FPR_o = []
roc_y_TPR_o = []

pr_x_fa = []
pr_y_fa = []
pr_x_fa2 = []
pr_y_fa2 = []
pr_x_o = []
pr_y_o = []

# y_true = gt_function_angle > 0.7
# y_true3 = gt_overlap_ > 0.3
# y_true = y_true3

# print('y_true', np.count_nonzero(y_true))
print('y_real_true', np.count_nonzero(y_real_true))
y_true = y_real_true

thres_range = np.concatenate((np.arange(0,1,0.05).reshape(-1,1), np.arange(0.98, 1.001, 0.001).reshape(-1,1)), axis=0).reshape(-1,)
for thres in thres_range:
    # y_true = gt_function_angle > thres
    y_test = model_outputs_function_angle > thres

    TP = np.count_nonzero(y_true & y_test)
    FP = np.count_nonzero(np.logical_not(y_true) & y_test)
    TN = np.count_nonzero(np.logical_not(y_true) & np.logical_not(y_test))
    FN = np.count_nonzero(y_true & np.logical_not(y_test))
    
    # TP2 = np.count_nonzero(y_true & y_test2)
    # FP2 = np.count_nonzero(np.logical_not(y_true) & y_test2)
    # TN2 = np.count_nonzero(np.logical_not(y_true) & np.logical_not(y_test2))
    # FN2 = np.count_nonzero(y_true & np.logical_not(y_test2))
    
    if (TP+FP) > 0 and (TP+FN) > 0:
        roc_x_FPR_fa.append(FP/(FP+TN)*100)
        roc_y_TPR_fa.append(TP/(TP+FN)*100)

        pr_x_fa.append(TP/(TP+FN)*100)
        pr_y_fa.append(TP/(TP+FP)*100)
    
    # if (TP2+FP2) > 0 and (TP2+FN2) > 0:
    #    roc_x_FPR_fa2.append(FP2/(FP2+TN2)*100)
    #    roc_y_TPR_fa2.append(TP2/(TP2+FN2)*100)

    #    pr_x_fa2.append(TP2/(TP2+FN2)*100)
    #    pr_y_fa2.append(TP2/(TP2+FP2)*100)

    # overlap
    # y_true2 = gt_overlap_ > thres
    y_test3 = model_outputs_overlap > thres

    TP3 = np.count_nonzero(y_true & y_test3)
    FP3 = np.count_nonzero(np.logical_not(y_true) & y_test3)
    TN3 = np.count_nonzero(np.logical_not(y_true) & np.logical_not(y_test3))
    FN3 = np.count_nonzero(y_true & np.logical_not(y_test3))
    
    if (TP3+FP3) > 0:
        roc_x_FPR_o.append(FP3/(FP3+TN3)*100)
        roc_y_TPR_o.append(TP3/(TP3+FN3)*100)

        pr_x_o.append(TP3/(TP3+FN3)*100)
        pr_y_o.append(TP3/(TP3+FP3)*100)
    

plt.rcParams.update({'font.size': 20})
plt.rcParams["figure.figsize"] = (8,6)

# ROC curve
plt.figure()
# plt.plot(roc_x_FPR_fa2, roc_y_TPR_fa2, marker='x', label='Proposed')
plt.plot(roc_x_FPR_fa, roc_y_TPR_fa, marker='x', label='Proposed')
plt.plot(roc_x_FPR_o, roc_y_TPR_o, marker='x', label='OverlapNet')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate [%]')
plt.ylabel('True Positive Rate [%]')
plt.xlim(0,105)
plt.ylim(0,105)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.2), ncol=2)
plt.tight_layout()
plt.savefig("pics/evaluation/roc_real_03.png")


# precision-recall curve# ROC curve
plt.figure()
# plt.plot(pr_x_fa2, pr_y_fa2, marker='x', label='Proposed')
plt.plot(pr_x_fa, pr_y_fa, marker='x', label='Proposed')
plt.plot(pr_x_o, pr_y_o, marker='x', label='OverlapNet')
plt.title('Precision-recall Curve')
plt.xlabel('Recall [%]')
plt.ylabel('Precision [%]')
plt.xlim(0,105)
plt.ylim(0,105)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.8), ncol=2)
plt.tight_layout()
plt.savefig("pics/evaluation/precision_recall_real_03.png")
