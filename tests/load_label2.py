from dl_ext.vision_ext.datasets.kitti.io import load_label_2

l = load_label_2('/home/linghao/Datasets/kitti', 'training', 4)
print(len(l))
print(l[0])
