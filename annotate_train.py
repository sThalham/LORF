import numpy as np
import json
import sys
import os
import transforms3d as tf3d
import csv
import cv2


fxkin = 572.41140
fykin = 573.57043
cxkin = 325.26110
cykin = 242.04899


def toPix_array(translation):

    xpix = ((translation[:, 0] * fxkin) / translation[:, 2]) + cxkin
    ypix = ((translation[:, 1] * fykin) / translation[:, 2]) + cykin
    #zpix = translation[2] * fxkin

    return np.stack((xpix, ypix), axis=1) #, zpix]


def draw_axis(img, poses):
    # unit is mm
    points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)

    rotMat = tf3d.quaternions.quat2mat(poses[3:7])
    rot, _ = cv2.Rodrigues(rotMat)
    tra = np.expand_dims(poses[0:3], axis=1) * 1000.0
    # [572.4114, 573.57043, 325.26110828, 242.04899594]
    K = np.float32([572.4114, 0., 325.26110828, 0., 573.57043, 242.04899594, 0., 0., 1.]).reshape(3,3)
    axisPoints, _ = cv2.projectPoints(points, rot, tra, K, (0, 0, 0, 0))

    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255,0,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0,255,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0,0,255), 3)
    return img


if __name__ == "__main__":

    train_dir = "/home/stefan/data/train_lorf/plate"
    annotations_target = os.path.join(train_dir, 'transforms_train.json')
    annotations_source = os.path.join(train_dir, 'groundtruth_pose.txt')
    associations_source = os.path.join(train_dir, 'associations.txt')
    markers_source = os.path.join(train_dir, 'groundtruth.txt')
    rgb_path = os.path.join(train_dir, 'rgb')
    #depth_path = os.path.join(train_dir, 'depth')

    tdbox_ori = np.array([[0.25, 0.0, 0.1],  # [35.087, 35.787, 60.686]
                                [0.25, 0.0, -0.05],
                                [0.25, -0.25, -0.05],
                                [0.25, -0.25, 0.1],
                                [-0.0, 0.0, 0.1],
                                [-0.0, 0.0, -0.05],
                                [-0.0, -0.25, -0.05],
                                [-0.0, -0.25, 0.1]])


    with open(associations_source, 'r') as f:
        reader = csv.reader(f, delimiter=" ")
        images = list(reader)
    ann_source = np.genfromtxt(annotations_source, delimiter=' ')
    mar_source = np.genfromtxt(markers_source, delimiter=' ')

    dict = {"camera_angle_x": 0.7505,
        "frames": []
    }

    rgb_imgs = os.listdir(rgb_path)
    #dep_imgs = os.listdir(depth_path)

    for idx in range(mar_source.shape[0]):

        img_path = os.path.join(".", images[idx][2][:-4])
        depth_img = cv2.imread(os.path.join(train_dir, images[idx][2]), -1)
        img = cv2.resize(depth_img, (640, 480))
        #print(depth_img.shape)
        #print(np.nanmax(depth_img))

        # annotations_pose
        cam_pose = np.eye((4))
        cam_pose[:3, 3] = mar_source[idx][1:4]# * 10.0
        quat = np.zeros((4))
        quat[1:] = mar_source[idx][4:-1]
        quat[0] = mar_source[idx][-1]
        cam_pose[:3, :3] = tf3d.quaternions.quat2mat(quat)

        tDbox = cam_pose[:3, :3].dot(tdbox_ori.T).T
        tDbox = tDbox + np.repeat(cam_pose[:3, 3][np.newaxis, :], 8, axis=0)
        box3D = toPix_array(tDbox)
        box3D = np.reshape(box3D, (16))
        #box3D = box3D.tolist()
        box3D = box3D.astype(np.uint16)

        #marker_pose = cam_pose
        #cam_pose = np.linalg.inv(cam_pose)
        #marker_pose = cam_pose

        #traquat = np.zeros((7))
        #traquat[:3] = marker_pose[:3, 3]
        #traquat[3:] = tf3d.quaternions.mat2quat(marker_pose[:3, :3])

        # annotations
        #marker_pose = np.array(ann_source[idx][1:]).reshape(4,4).T
        #marker_pose = np.linalg.inv(marker_pose)
        #print(marker_pose)

        traquat = np.zeros((7))
        traquat[:3] = cam_pose[:3, 3]
        traquat[3:] = tf3d.quaternions.mat2quat(cam_pose[:3, :3])

        colR = 250
        colG = 25
        colB = 175
        '''
        #rot_lie = [[0.0, pose[3], pose[4]], [-pose[3], 0.0, pose[5]], [-pose[4], -pose[5], 0.0]]
        #ssm =np.asarray(rot_lie, dtype=np.float32)
        #map = geometry.rotations.map_hat(ssm)
        #quat = tf3d.euler.euler2quat(pose2D[3], pose2D[4], pose2D[5])
        #quat = quat / np.linalg.norm(quat)
        pose2D = np.concatenate([postra[i], pose2D[3:]])
        #print('x: ', (pose[0]-bb[2])/(bb[4]-bb[2]))
        #print('y: ', (pose[1] - bb[1]) / (bb[3] - bb[1]))
        #print(pose[0:2], bb[1:])

        #cv2.circle(img, (int(pose[0]), int(pose[1])), 5, (0, 255, 0), 3)
        draw_axis(aug_xyz, pose2D)
        '''
        pose = box3D

        img = cv2.line(img, tuple(pose[0:2].ravel()), tuple(pose[2:4].ravel()), (130, 245, 13), 2)
        img = cv2.line(img, tuple(pose[2:4].ravel()), tuple(pose[4:6].ravel()), (50, 112, 220), 2)
        img = cv2.line(img, tuple(pose[4:6].ravel()), tuple(pose[6:8].ravel()), (50, 112, 220), 2)
        img = cv2.line(img, tuple(pose[6:8].ravel()), tuple(pose[0:2].ravel()), (50, 112, 220), 2)
        img = cv2.line(img, tuple(pose[0:2].ravel()), tuple(pose[8:10].ravel()), (colR, colG, colB), 2)
        img = cv2.line(img, tuple(pose[2:4].ravel()), tuple(pose[10:12].ravel()), (colR, colG, colB), 2)
        img = cv2.line(img, tuple(pose[4:6].ravel()), tuple(pose[12:14].ravel()), (colR, colG, colB), 2)
        img = cv2.line(img, tuple(pose[6:8].ravel()), tuple(pose[14:16].ravel()), (colR, colG, colB), 2)
        img = cv2.line(img, tuple(pose[8:10].ravel()), tuple(pose[10:12].ravel()), (colR, colG, colB), 2)
        img = cv2.line(img, tuple(pose[10:12].ravel()), tuple(pose[12:14].ravel()), (colR, colG, colB), 2)
        img = cv2.line(img, tuple(pose[12:14].ravel()), tuple(pose[14:16].ravel()), (colR, colG, colB), 2)
        img = cv2.line(img, tuple(pose[14:16].ravel()), tuple(pose[8:10].ravel()), (colR, colG, colB), 2)


        draw_axis(img, traquat)
        # cv2.imwrite("/home/stefan/LORF_viz/depth_" + str(idx) + ".png", (depth_img * (255/np.nanmax(depth_img))).astype(np.uint8))
        cv2.imwrite("/home/stefan/LORF_viz/depth_" + str(idx) + ".png", img)

        #cam_pose = np.linalg.inv(ann_source[idx, 1:].reshape(4,4).T)
        #cam_pose = ann_source[idx, 1:].reshape(4, 4).T
        #cam_pose = np.eye((4))
        #print(mar_source[idx], len(mar_source[idx]))
        #print(mar_source[idx])
        #cam_pose[:3, 3] = mar_source[idx][1:4]
        #cam_pose[:3, :3] = tf3d.quaternions.quat2mat(mar_source[idx][4:])
        print(idx, 'cam_pose: ', cam_pose[:3, 3])
        cam_pose = cam_pose.tolist()
        current_view = {"file_path": img_path,
                        "rotation" : 0.0,
                        "transform_matrix" : cam_pose
        }
        dict["frames"].append(current_view)

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