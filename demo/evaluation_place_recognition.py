import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import f1_score 
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/utils'))
from utils import *
import json

# 
seq = "06"
d_thres = "4"

# load function angle result
# prediction_function_angle = np.empty((0,4))
# test_result_filenames = ["/home/cel/DockerFolder/code/overlapnet_function_angle/result/seq00_test_function_angle_ell_1_4000k/validation_results.npz",
#                         "/home/cel/DockerFolder/code/overlapnet_function_angle/result/seq00_test_function_angle_ell_1_4000k_end/validation_results.npz",
#                         ]

# for test_result_filename in test_result_filenames:
#     with np.load(test_result_filename) as data:
#         prediction_function_angle = np.vstack((prediction_function_angle, data['arr_0.npy']))


# fa_data_ell_03 = np.load("/home/cel/DockerFolder/data/kitti/sequences/00/ground_truth_function_angle/ground_truth_overlap_yaw.npz")
# prediction_function_angle2 = fa_data_ell_03['overlaps']

# load overlapnet result
# prediction_overlap = np.empty((0,4))
# test_result_filenames = ["/home/cel/DockerFolder/code/overlapnet_function_angle/result/seq00_test_overlap_5000k/validation_results.npz",
#                         "/home/cel/DockerFolder/code/overlapnet_function_angle/result/seq00_test_overlap_5000k_end/validation_results.npz",
#                         ]
# # test_result_filenames = ["/home/cel/DockerFolder/code/overlapnet_function_angle/result/seq00_test_all/seq00_test_overlap_2000k/validation_results.npz",
# #                         "/home/cel/DockerFolder/code/overlapnet_function_angle/result/seq00_test_all/seq00_test_overlap_2000k_4000k/validation_results.npz",
# #                         "/home/cel/DockerFolder/code/overlapnet_function_angle/result/seq00_test_all/seq00_test_overlap_4000k_6000k/validation_results.npz",
# #                         "/home/cel/DockerFolder/code/overlapnet_function_angle/result/seq00_test_all/seq00_test_overlap_6000k_8000k/validation_results.npz",
# #                         "/home/cel/DockerFolder/code/overlapnet_function_angle/result/seq00_test_all/seq00_test_overlap_8000k_end/validation_results.npz"           
# #                         ]
# for test_result_filename in test_result_filenames:
#     with np.load(test_result_filename) as data:
#         prediction_overlap = np.vstack((prediction_overlap, data['arr_0.npy']))

# Load ground truth

fa_data_ell_1 = np.load("/home/cel/DockerFolder/data/kitti/sequences/"+seq+"/ground_truth_function_angle/ground_truth_overlap_yaw.npz")
prediction_function_angle = fa_data_ell_1['overlaps']

overlap_data_oxford = np.load("/home/cel/DockerFolder/data/oxford_test/2014-12-09-13-21-02/ground_truth_overlap/ground_truth_overlap_yaw.npz")
prediction_overlap_oxford = overlap_data_oxford['overlaps']
print('prediction_overlap_oxford datasize:', prediction_overlap_oxford.shape[0])

overlap_data = np.load("/home/cel/DockerFolder/data/kitti/sequences/"+seq+"/ground_truth_overlap/ground_truth_overlap_yaw.npz")
prediction_overlap = overlap_data['overlaps']

print('function angle datasize:', prediction_function_angle.shape[0])
print('overlap datasize:', prediction_overlap.shape[0])
# print('function angle ell 1 datasize', prediction_function_angle.shape[0], 'overlap datasize', prediction_overlap.shape[0])
# print('\nfunction angle data (ell=1.0)\n', prediction_function_angle[:10, :], prediction_function_angle[-10:, :])
# print('\nfunction angle data (ell=0.3)\n', prediction_function_angle2[:20, :])
# print('\noverlap data\n', prediction_overlap[:10, :], prediction_overlap[-10:, :])
overlap_test_imgf1 = prediction_overlap[:,0]
overlap_test_imgf2 = prediction_overlap[:,1]
model_outputs_overlap = prediction_overlap[:,2]

function_angle_test_imgf1 = prediction_function_angle[:,0]
function_angle_test_imgf2 = prediction_function_angle[:,1]
model_outputs_function_angle = prediction_function_angle[:,2]


# model_outputs_function_angle2 = prediction_function_angle2[:,2]


# pointnetvlad
pointnetvlad_global_features = np.load("/home/cel/DockerFolder/code/pointnetvlad_kxhit/exp_results/log_fold"+seq+"/feature_database/fov100/"+seq+"_PV_"+seq+".npy")
# pointnetvlad_global_features = np.load("/home/cel/DockerFolder/code/pointnetvlad_kxhit/pretrained_results/feature_database/fov100/00_PV_ref.npy")
print('pointnetvlad_global_features', pointnetvlad_global_features.shape)

# get ground truth place recognition
f = open("/home/cel/DockerFolder/code/pointnetvlad_kxhit/kitti_results/"+seq+"_triplets/positive_sequence_D-"+d_thres+"_T-0.json",)
positive_sequence = json.load(f)
positive_sequence_seq = positive_sequence[seq]
f.close()

y_true_all = []
index_mask_not_near = []
y_pointnetvlad = []
for pair_index in range(len(function_angle_test_imgf1)):
    query_frame_index = int(function_angle_test_imgf1[pair_index])
    database_frame_index = int(function_angle_test_imgf2[pair_index])

    if database_frame_index < (query_frame_index - 50):
        index_mask_not_near.append(int(pair_index))

    y_pointnetvlad.append(np.dot(pointnetvlad_global_features[query_frame_index, :], pointnetvlad_global_features[database_frame_index, :]))

    # obtain true loops
    true_neighbors = positive_sequence_seq[str(query_frame_index)]

    # only take scans before current scan - 50
    true_neighbors_mask = [i for i in true_neighbors if i < query_frame_index-50]

    if database_frame_index in true_neighbors_mask:
        y_true_all.append(1)
    else:
        y_true_all.append(0)


print('total number of test pairs:', len(y_true_all))
print('y_true_all non zeros:', np.count_nonzero(y_true_all))
print('index_mask_not_near:', len(index_mask_not_near))
y_true_all = np.array(y_true_all)
y_true = y_true_all[index_mask_not_near]
y_pointnetvlad = np.array(y_pointnetvlad)


# save precompute files
np.save("/home/cel/DockerFolder/data/kitti/sequences/"+seq+"/loop_clousre_gt_D"+d_thres+".npy", np.array(y_true_all))
np.save("/home/cel/DockerFolder/data/kitti/sequences/"+seq+"/index_mask_not_near.npy", np.array(index_mask_not_near))
np.save("/home/cel/DockerFolder/data/kitti/sequences/"+seq+"/pointnetvlad.npy", y_pointnetvlad)

# load precompute files
# y_true_all = np.load('/home/cel/DockerFolder/data/kitti/sequences/00/loop_clousre_gt.npy')
# index_mask_not_near = np.load('/home/cel/DockerFolder/data/kitti/sequences/00/index_mask_not_near.npy')
# y_true = y_true_all[index_mask_not_near]


# mask results, only check not near frames
y_overlapnet = model_outputs_overlap[index_mask_not_near]
y_functionangle = model_outputs_function_angle[index_mask_not_near]
# y_functionangle2 = model_outputs_function_angle2[index_mask_not_near]
y_pointnetvlad = y_pointnetvlad[index_mask_not_near]


# Calculate precision and recall using scipy library function precision_recall_curve()
precision_overlapnet, recall_overlapnet, thresholds_overlapnet = precision_recall_curve(y_true, y_overlapnet)
precision_functionangle, recall_functionangle, thresholds_functionangle = precision_recall_curve(y_true, y_functionangle)
# precision_functionangle2, recall_functionangle2, thresholds_functionangle2 = precision_recall_curve(y_true, y_functionangle2)
precision_pointnetvlad, recall_pointnetvlad, thresholds_pointnetvlad = precision_recall_curve(y_true, y_pointnetvlad)


# Plot Precision-recall curve
plt.figure()
plt.plot(recall_functionangle*100, precision_functionangle*100, '-', label='Proposed GT')
# plt.plot(recall_functionangle2*100, precision_functionangle2*100, '-.', label='Proposed (ell=0.3) GT')
plt.plot(recall_overlapnet*100, precision_overlapnet*100, '--', label='OverlapNet GT')
plt.plot(recall_pointnetvlad*100, precision_pointnetvlad*100, ':', label='PointNetVLAD')
plt.title("KITTI "+seq+" Precision-recall Curve (D"+d_thres+")")
plt.xlabel('Recall [%]')
plt.ylabel('Precision [%]')
plt.xlim(0,105)
plt.ylim(0,105)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.2), ncol=3)
plt.tight_layout()
plt.savefig("pics/evaluation/precision_recall_seq"+seq+"_D"+d_thres+"_gt.png")
print('Precision-recall curve is saved at:', "pics/evaluation/precision_recall_seq"+seq+"_D"+d_thres+"_gt.png")


# # Calculate F1 Score using scipy library function f1_score()
# f1_overlapnet = f1_score(y_true, y_overlapnet, average='binary')
# f1_functionangle = f1_score(y_true, y_functionangle, average='binary')
# f1_pointnetvlad = f1_score(y_true, y_pointnetvlad, average='binary')
# # print results
# print('Place Recognition Result - F1 Score:')
# print('OverlapNet:', f1_overlapnet)
# print('Function Angle:', f1_functionangle)
# print('PointNetVLAD:', f1_pointnetvlad)