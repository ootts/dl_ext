from dl_ext.vision_ext.datasets.kitti.load import load_label_2

l = load_label_2('/home/linghao/Datasets/kitti', 'training', 4)
print(len(l))
