from dl_ext.vision_ext.datasets.kitti.io.velodyne import *

root = '/home/linghao/Datasets/kitti'
print(load_velodyne(root, 'training', 3).shape)
# print(load_velodyne(root, 'training', '000003').shape)
# print(load_velodyne(os.path.join(root,'training','000003.bin')))
