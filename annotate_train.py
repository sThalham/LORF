import numpy as np
import json
import sys
import os
import transforms3d as tf3d
import csv


if __name__ == "__main__":

    train_dir = "/home/stefan/data/train_lorf/plate"
    annotations_target = os.path.join(train_dir, 'transforms_train.json')
    annotations_source = os.path.join(train_dir, 'groundtruth_pose.txt')
    associations_source = os.path.join(train_dir, 'associations.txt')
    markers_source = os.path.join(train_dir, 'groundtruth.txt')

    with open(associations_source, 'r') as f:
        reader = csv.reader(f, delimiter=" ")
        images = list(reader)
    ann_source = np.genfromtxt(annotations_source, delimiter=' ')
    mar_source = np.genfromtxt(markers_source, delimiter=' ')

    dict = {"camera_angle_x": 0.7505,
        "frames": []
    }

    for idx in range(mar_source.shape[0]):
        img_path = os.path.join(".", images[idx][2][:-4])
        #print(img_path)
        #cam_pose = np.linalg.inv(ann_source[idx, 1:].reshape(4,4).T)
        #cam_pose = ann_source[idx, 1:].reshape(4, 4).T
        cam_pose = np.eye((4))
        print(mar_source[idx], len(mar_source[idx]))
        print(mar_source[idx][1:4])
        cam_pose[:3, 3] = mar_source[idx][1:4]
        cam_pose[:3, :3] = tf3d.quaternions.quat2mat(mar_source[idx][4:])
        print('cam_pose: ', cam_pose)
        cam_pose = cam_pose.tolist()
        current_view = {"file_path": img_path,
                        "rotation" : 0.0,
                        "transform_matrix" : cam_pose
        }
        dict["frames"].append(current_view)

    print

    #f = open(annotations_source, 'r')
    #content = f.read()
    #print(content[0])
    #print(content[1])
    #print(content[2])
    #poses = []
    #for idx in range(len(content)):
    #    poses.append(content[idx:(idx+11)])
    #    idx += 11

    #print(poses)


    with open(annotations_target, 'w') as fpT:
        json.dump(dict, fpT)