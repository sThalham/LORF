import numpy as np
import json
import sys
import os
import transforms3d as tf3d
import csv
import cv2


<<<<<<< HEAD
fx = 914.79047
fy = 915.3621966666
cx = 632.46826666666
cy = 374.607362
=======
fxkin = 572.41140
fykin = 573.57043
cxkin = 325.26110
cykin = 242.04899


def toPix_array(translation):

    xpix = ((translation[:, 0] * fxkin) / translation[:, 2]) + cxkin
    ypix = ((translation[:, 1] * fykin) / translation[:, 2]) + cykin
    #zpix = translation[2] * fxkin

    return np.stack((xpix, ypix), axis=1) #, zpix]
>>>>>>> 48f8450469adb620f48355fd3c3724aa247155ef


def draw_axis(img, poses):
    # unit is mm
    points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)

    rotMat = tf3d.quaternions.quat2mat(poses[3:7])
    rot, _ = cv2.Rodrigues(rotMat)
    tra = np.expand_dims(poses[0:3], axis=1) * 1000.0
    # [572.4114, 573.57043, 325.26110828, 242.04899594]
    K = np.float32([fx, 0., cx, 0., fy, cy, 0., 0., 1.]).reshape(3,3)
    axisPoints, _ = cv2.projectPoints(points, rot, tra, K, (0, 0, 0, 0))

    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255,0,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0,255,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0,0,255), 3)
    return img


if __name__ == "__main__":

    train_dir = "/home/stefan/data/train_lorf/plate"
    target = "/home/stefan/data/train_lorf/plate_robot_pose"
    annotations_target = os.path.join(target, 'transforms_train.json') # target for nerf
    annotations_source = os.path.join(train_dir, 'groundtruth_pose.txt') # cam pose in robot
    associations_source = os.path.join(train_dir, 'associations.txt')
    markers_source = os.path.join(train_dir, 'groundtruth.txt') # marker pose in cam
    empty_source = os.path.join(train_dir, 'empty_groundtruth.txt')
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
    emp_source = np.genfromtxt(empty_source, delimiter=' ')

    width = 1280
    height = 720
    # fx = 923.101
    # fy = 922.568
    # cx = 629.3134765625
    # cy = 376.28814697265625
    fx = 914.79047
    fy = 915.3621966666
    cx = 632.46826666666
    cy = 374.607362

    dict = {"camera_angle_x": 0.7505,
        "frames": []
    }

    rgb_imgs = os.listdir(rgb_path)

    cam_init = np.eye((4))
    cam_init[:3, 3] = mar_source[0][1:4]
    quat = np.zeros((4))
    quat[1:] = mar_source[0][4:-1]
    quat[0] = mar_source[0][-1]
    cam_init[:3, :3] = tf3d.quaternions.quat2mat(quat)

    rob_init = ann_source[0][1:].reshape((4, 4)).T
    print(rob_init)
    #rob_init[:3, 3] = ann_source[0][1:4]
    #quat = np.zeros((4))
    #quat[1:] = ann_source[0][4:-1]
    #quat[0] = ann_source[0][-1]
    #rob_init[:3, :3] = tf3d.quaternions.quat2mat(quat)

    for idx in range(emp_source.shape[0]):
        img_path = os.path.join(".", images[idx][2][:-4])
        img = cv2.imread(os.path.join(train_dir, images[idx][2]), -1)
        # png = png[:, 420:1140, :]
        img = img[int(cy - 320):int(cy + 320), int(cx - 320):int(cx + 320), :]
        img_store = os.path.join(target, images[idx][2])

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

        #cam_pose = np.linalg.inv(ann_source[idx, 1:].reshape(4,4).T)
        #cam_pose = ann_source[idx, 1:].reshape(4, 4).T
        #quat = np.zeros((4))
        #quat[1:] = mar_source[idx][4:-1]
        #quat[0] = mar_source[idx][-1]
        #cam_pose[:3, :3] = tf3d.quaternions.quat2mat(quat)
        #cam_pose = np.linalg.inv(cam_pose)
        #print(cam_pose)

        rob_pose = ann_source[idx][1:].reshape((4, 4)).T
        in_robot_diff = np.matmul(np.linalg.inv(rob_pose), rob_init)
        cam_pose = np.matmul(cam_init, in_robot_diff)
        cam_pose = np.linalg.inv(cam_pose)

        # visualization
        traquat = np.zeros((7))
        traquat[:3] = cam_pose[:3, 3]
        traquat[3:] = tf3d.quaternions.mat2quat(cam_pose[:3, :3])
        draw_axis(img, traquat)

        cam_pose = cam_pose.tolist()
        current_view = {"file_path": img_store[:-4],
                        "rotation" : 0.0,
                        "transform_matrix" : cam_pose
        }
        dict["frames"].append(current_view)
        cv2.imwrite(img_store, img)

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