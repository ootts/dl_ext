import os
from multipledispatch import dispatch
import numpy as np


class Calibration:
    def __init__(self, calibs):
        self.P0 = calibs['P0']  # 3 x 4
        self.P1 = calibs['P1']  # 3 x 4
        self.P2 = calibs['P2']  # 3 x 4
        self.R0 = calibs['R0_rect']  # 3 x 3
        self.V2C = calibs['Tr_velo_to_cam']  # 3 x 4
        self.I2V = calibs['Tr_imu_to_velo']  # 3 x 4

        # Camera intrinsics and extrinsics
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)

    def cart_to_hom(self, pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def lidar_to_rect(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        pts_rect = np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))
        # pts_rect = reduce(np.dot, (pts_lidar_hom, self.V2C.T, self.R0.T))
        return pts_rect

    def rect_to_img(self, pts_rect):
        """
        :param pts_rect: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        pts_rect_depth = pts_2d_hom[:, 2] - self.P2.T[3, 2]  # depth in rect camera coord
        return pts_img, pts_rect_depth

    def lidar_to_img(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect = self.lidar_to_rect(pts_lidar)
        pts_img, pts_depth = self.rect_to_img(pts_rect)
        return pts_img, pts_depth


@dispatch(str)
def load_calib(absolute_path):
    with open(absolute_path) as f:
        lines = {line.strip().split(':')[0]: list(map(float, line.strip().split(':')[1].split())) for line in
                 f.readlines()[:-1]}
    calibs = {}
    calibs['P0'] = np.array(lines['P0']).reshape((3, 4))
    calibs['P1'] = np.array(lines['P1']).reshape((3, 4))
    calibs['P2'] = np.array(lines['P2']).reshape((3, 4))
    # R0_rect = np.eye(4)
    # R0_rect[0:3, 0:3] = np.array(lines['R0_rect']).reshape((3, 3))
    calibs['R0_rect'] = np.array(lines['R0_rect']).reshape((3, 3))
    # Tr_velo_to_cam = np.eye(4)
    # Tr_velo_to_cam[0:3, :] = np.array(lines['Tr_velo_to_cam']).reshape((3, 4))
    calibs['Tr_velo_to_cam'] = np.array(lines['Tr_velo_to_cam']).reshape((3, 4))
    # Tr_imu_to_velo = np.eye(4)
    # Tr_imu_to_velo[0:3, :] = np.array(lines['Tr_imu_to_velo']).reshape((3, 4))
    calibs['Tr_imu_to_velo'] = np.array(lines['Tr_imu_to_velo']).reshape((3, 4))
    return Calibration(calibs)


@dispatch(str, str)
def load_calib(calib_dir, imgid):
    absolute_path = os.path.join(calib_dir, imgid + '.txt')
    return load_calib(absolute_path)


@dispatch(str, int)
def load_calib(calib_dir, imgid):
    imgid = '%06d' % imgid
    return load_calib(calib_dir, imgid)


@dispatch(str, str, str)
def load_calib(kitti_root, split, imgid):
    calib_dir = os.path.join(kitti_root, 'object', split, 'calib')
    return load_calib(calib_dir, imgid)


@dispatch(str, str, int)
def load_calib(kitti_root, split, imgid):
    imgid = '%06d' % imgid
    return load_calib(kitti_root, split, imgid)
