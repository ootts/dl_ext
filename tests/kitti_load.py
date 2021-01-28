from vis3d import Vis3D

from dl_ext.vision_ext.datasets.kitti.io import *
import dl_ext

print('dl-ext version', dl_ext.__version__)
KITTIROOT = osp.expanduser('~/Datasets/kitti')
imgid = 3
img2 = load_image_2(KITTIROOT, 'training', imgid)
img3 = load_image_3(KITTIROOT, 'training', imgid)
calib = load_calib(KITTIROOT, 'training', imgid)
label2 = load_label_2(KITTIROOT, 'training', imgid)
label3 = load_label_3(KITTIROOT, 'training', imgid)
lidar = load_velodyne(KITTIROOT, 'training', imgid)[:, :3]
h, w, _ = load_image_info(KITTIROOT, 'training', imgid)

pts_rect = calib.lidar_to_rect(lidar)
keep = calib.filter_fov_pts(pts_rect)
pts_rect = pts_rect[keep]
with Vis3D(
        xyz_pattern=('x', '-y', '-z'),
        out_folder="dbg"
) as vis3d:
    vis3d.set_scene_id(imgid)
    vis3d.add_point_cloud(pts_rect)
