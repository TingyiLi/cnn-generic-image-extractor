# coding: utf-8

from PIL import Image
import numpy as np
import nrrd
import os


def getLesionSilces(nrrd_path, nrrd_seg_path):
    # 读取文件
    nrrd_data, nrrd_options = nrrd.read(nrrd_path)
    label_data, label_options = nrrd.read(nrrd_seg_path)
    # 匹配位置
    s1 = label_options['keyvaluepairs']['Segmentation_ReferenceImageExtentOffset']
    s2 = s1.split(' ')
    start_position = {}
    start_position['x'] = int(s2[0])
    start_position['y'] = int(s2[1])
    start_position['z'] = int(s2[2])
    # sizes
    label_size = label_options['sizes']
    # pipeilabel区域
    st_x = start_position['x']
    ed_x = start_position['x'] + label_size[0]
    st_y = start_position['y']
    ed_y = start_position['y'] + label_size[1]
    st_z = start_position['z']
    ed_z = start_position['z'] + label_size[2]
    data_in_label = nrrd_data[st_x:ed_x, st_y:ed_y, st_z:ed_z]
    return data_in_label


def select_max_lesion(data_in_label):
    shape = data_in_label.shape
    areas = []
    for cnt in range(shape[2]):
        sample_label_data = data_in_label[:, :, cnt]
        sample_label_data_copy = sample_label_data.copy()
        sample_label_data_copy[np.where(np.int8(np.logical_not(np.logical_not(sample_label_data_copy))))] = 1
        areas.append(np.sum(sample_label_data_copy))
    id_ = np.where(areas == np.max(areas))
    max_lesion = data_in_label[:, :, int(id_[0][0])]
    return max_lesion


def get_one_sample_result(nrrd_path, nrrd_seg_path, outpath, id):
    data_in_label = getLesionSilces(nrrd_path, nrrd_seg_path)
    max_lesion = select_max_lesion(data_in_label)
    img = Image.fromarray(max_lesion).convert('L')
    img.save(os.path.join(outpath, id + ".png"))


if __name__ == "__main__":
    nrrd_path = r'C:\Users\tingyi\Desktop\work\S729\nrrd\dealed\001.nrrd'  # nrrd 地址
    nrrd_seg_path = r'C:\Users\tingyi\Desktop\work\S729\nrrd\dealed\001-label.seg.nrrd'  # need segmentation 地址
    out_path = r'C:\Users\tingyi\Desktop\work\S729\nrrd\image'  # 保存地址
    id = '001'  # 输出的图片名称
    get_one_sample_result(nrrd_path, nrrd_seg_path, out_path, id)
