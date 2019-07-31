# coding: utf-8
import pydicom as pdcm
from PIL import Image
import os
import numpy as np

"""
处理Dicom文件：从dicom序列中提取出数据并匹配病灶区，导出最大病灶取数据
"""


class ParseDICOM:
    def __init__(self, dcms_path, rt_path, out_path):
        self.__dcms_path = dcms_path
        self.__rt_path = rt_path
        self.__out_path = out_path

    def __util_read_dcms(self):
        dcm_files = os.listdir(self.__dcms_path)  # 遍历文件夹内的所有文件
        dcms_data = {}  # 以{UID:Data}形式返回结果
        for dcm in dcm_files:  # 遍历读取数据
            dcm_path = self.__dcms_path + '/' + dcm
            ds = pdcm.dcmread(dcm_path)
            data = ds.pixel_array
            dcm_origin = ds.ImagePositionPatient  # 网格原点在世界坐标系的位置
            dcm_spacing = ds.PixelSpacing  # 采样间隔
            uid = ds.SOPInstanceUID
            dcms_data[uid] = {'dcmSpacing': dcm_spacing, 'dcmOrigin': dcm_origin, 'data': data}
        # 返回结果
        return dcms_data

    def __util_read_rtdcm(self):
        rt_rdata = {}  # 以{RUID:label_data}形式返回结果
        rt_ds = pdcm.dcmread(self.__rt_path)
        sequences = rt_ds.ROIContourSequence[0].ContourSequence
        for sequence in sequences:
            RUID = sequence.ContourImageSequence[0].ReferencedSOPInstanceUID
            Label_data = sequence.ContourData
            num = sequence.NumberOfContourPoints
            rt_rdata[RUID] = {'pointNumber': num, 'data': Label_data}
        # 返回结果
        return rt_rdata

    def __util_convert_global_aix_to_net_pos(self, rt_data, dcms_data):
        point_data = {}  # 返回坐标{uid:data}
        for rt_key, rt_value in rt_data.items():
            num = rt_value['pointNumber']
            label_data = rt_value['data']
            dcm_origin = dcms_data[rt_key]['dcmOrigin']
            dcm_spacing = dcms_data[rt_key]['dcmSpacing']
            point = []  # 坐标[(x1,y1),(...),...]
            for jj in range(num):
                ii = jj * 3
                x = label_data[ii]  # 轮廓世界坐标系
                y = label_data[ii + 1]
                X = int(float(x) - float(dcm_origin[0]) / float(dcm_spacing[0]))  # 轮廓X坐标
                Y = int(float(y) - float(dcm_origin[1]) / float(dcm_spacing[1]))  # 轮廓Y坐标
                point.append((X, Y))
            point_data[rt_key] = point
        return point_data

    def __util_fill_label_inner(self, label_mat):
        shape = label_mat.shape
        S = label_mat
        st = ed = 0
        for i in range(shape[0]):  # 填充行
            if sum(label_mat[i, :]) >= 2:
                for k in range(shape[1]):  # 找起点
                    if S[i, k] > 0:
                        st = k
                        break
                for k in range(shape[1]):  # 找终点
                    if S[i, shape[1] - 1 - k] > 0:
                        ed = shape[1] - 1 - k
                        break
                S[i, st:ed] = 1
        for j in range(shape[1]):  # 填充列
            if sum(label_mat[:, j]) >= 2:
                for k in range(shape[0]):  # 找起点
                    if S[k, j] > 0:
                        st = k
                        break
                for k in range(shape[0]):  # 找终点
                    if S[shape[1] - 1 - k, j] > 0:
                        ed = shape[1] - 1 - k
                        break
                S[st:ed, j] = 1
        return S

    def __util_match_label_dcms(self, dcms_data, point_data):
        des_data = {}  # 返回数据：{uid:{dcm-data,ROI-data}}
        for uid, points in point_data.items():
            dcm_data = dcms_data[uid]['data']
            points = point_data[uid]
            x = []
            y = []
            sz = dcm_data.shape
            S = np.zeros((sz[0], sz[1]))
            # 构造label图像
            for point in points:
                x.append(point[0])
                y.append(point[1])
                S[x, y] = 1
            # 转置
            S = np.transpose(S)
            # 填充label内部
            S = self.__util_fill_label_inner(S)
            # 从dcm中截取label区域
            ROI_data = dcm_data * S
            des_data[uid] = {"DCMData": dcm_data, "ROIData": ROI_data}
        return des_data

    def __get_ROI_region_in_dcms(self):
        # 读取数据
        dcms_data = self.__util_read_dcms()
        rt_data = self.__util_read_rtdcm()
        # 坐标转换
        point_data = self.__util_convert_global_aix_to_net_pos(rt_data, dcms_data)
        # 匹配label抽取数据
        sample_data = self.__util_match_label_dcms(dcms_data, point_data)
        return sample_data

    def util_rerange_mat(self, mat, MIN, MAX):
        mat = mat.astype(np.float32)
        MAX_MAT = np.max(mat)
        MIN_MAT = np.min(mat)
        mat = mat - MIN_MAT + MIN
        if not MAX_MAT == 0:
            mat = mat / MAX_MAT * MAX
        return mat

    def __util_select_max_lesion(self, imgs_data):
        areas = []
        for img in imgs_data:
            img_cp = img.copy()
            img_cp[np.where(np.int8(np.logical_not(np.logical_not(img_cp))))] = 1
            areas.append(np.sum(np.sum(img_cp)))
        id_ = np.where(areas == np.max(areas))
        return int(id_[0])

    def __util_crop_ROI_AND_resize_to_fixed(self, img_data, sz_w, sz_h, threshold):
        # 裁剪ROI区域
        shape = img_data.shape
        rst = red = cst = ced = 0
        # MIN = np.min(img_data)
        MIN = 0
        img_mask = (img_data > MIN).astype(np.int)
        for i in range(shape[0]):
            if np.sum(img_mask[i, :]) > threshold:
                rst = i  # 起始行
                break
        for i in range(shape[0]):
            if np.sum(img_mask[shape[0] - 1 - i, :]) > threshold:
                red = shape[0] - 1 - i  # 终止行
                break
        for i in range(shape[1]):
            if np.sum(img_mask[:, i]) > threshold:
                cst = i  # 起始列
                break
        for i in range(shape[1]):
            if np.sum(img_mask[:, shape[1] - 1 - i]) > threshold:
                ced = shape[1] - 1 - i  # 终止列
                break
        if rst >= red or cst >= ced:
            raise (Exception("截取感兴趣区域出错！"))
        ROI = img_data[rst:red, cst:ced]
        try:
            ROI_RESIZED_IMG = Image.fromarray(ROI).resize((sz_w, sz_h), Image.ANTIALIAS)
        except:
            raise (Exception("矩阵缩放出错！"))
        return ROI_RESIZED_IMG

    def ct_rt_extractor(self):
        try:
            des_data = self.__get_ROI_region_in_dcms()
        except:
            raise (Exception("从CT序列匹配病灶出错！"))
        # 输出图片
        imgs_data = []
        ID = []
        for _id, data in des_data.items():
            ROI_data = data['ROIData']
            imgs_data.append(ROI_data)
            ID.append(_id)
        # 选择最大病灶切片
        try:
            id_ = self.__util_select_max_lesion(imgs_data)
        except:
            raise (Exception("选择最大病灶区出错！"))
        # 截取最大病灶区域
        try:
            sz_w = 299
            sz_h = 299
            MAX_ROI_RESIZED_IMG = self.__util_crop_ROI_AND_resize_to_fixed(imgs_data[id_], sz_w, sz_h, 0)
        except:
            raise (Exception("截取病灶区及重采样出错！"))
        return MAX_ROI_RESIZED_IMG, des_data[ID[id_]]


def parserForBatchDicom(dcms_rt_path, out_path):
    # 注意文件夹结构必须类似如下：
    # 一级：LUNG
    # 二级：LUNG-1,LUNG-2,...,
    # 三级：09-18-2008-StudyID-69331
    # 四级：0-82046,0-95085
    # 五级：01.dcm,02.dcm,...,
    folders = os.listdir(dcms_rt_path)
    cnt = 1
    for sample in folders:
        IDs = os.listdir(dcms_rt_path + '/' + sample)
        dcms_path = rt_path = ''
        for id_ in IDs:
            ct_rt = os.listdir(dcms_rt_path + '/' + sample + '/' + id_)
            if len(ct_rt) < 2:
                print(sample, " 缺少文件，跳过！")
                continue
            else:
                if len(os.listdir(dcms_rt_path + '/' + sample + '/' + id_ + '/' + ct_rt[0])) > len(
                        os.listdir(dcms_rt_path + '/' + sample + '/' + id_ + '/' + ct_rt[1])):
                    dcms_path = dcms_rt_path + '/' + sample + '/' + id_ + '/' + ct_rt[0]
                    file = os.listdir(dcms_rt_path + '/' + sample + '/' + id_ + '/' + ct_rt[1])
                    rt_path = dcms_rt_path + '/' + sample + '/' + id_ + '/' + ct_rt[1] + '/' + file[0]
                else:
                    file = os.listdir(dcms_rt_path + '/' + sample + '/' + id_ + '/' + ct_rt[0])
                    rt_path = dcms_rt_path + '/' + sample + '/' + id_ + '/' + ct_rt[0] + '/' + file[0]
                    dcms_path = dcms_rt_path + '/' + sample + '/' + id_ + '/' + ct_rt[1]
        # 抽取数据
        parser = ParseDICOM(dcms_path=dcms_path, rt_path=rt_path, out_path=out_path)
        MAX_ROI_RESIZED_IMG, ORIGIN = parser.ct_rt_extractor()
        print(cnt, ': ', sample)
        cnt += 1
        if MAX_ROI_RESIZED_IMG is False:
            print("当前文件出错，跳过！")
        else:
            MAX_ROI_RESIZED_IMG = MAX_ROI_RESIZED_IMG.convert('L')
            MAX_ROI_RESIZED_IMG.save(out_path + '/' + sample + '.png', 'png')

            ORIGIN_CT_IMG = Image.fromarray((ORIGIN['DCMData'])).convert('L')
            ORIGIN_ROI_IMG = Image.fromarray((ORIGIN['ROIData'])).convert('L')
            ORIGIN_CT_IMG.save(out_path + '/' + sample + '-ORIGIN' + '.png', 'png')
            ORIGIN_ROI_IMG.save(out_path + '/' + sample + '-ORIGIN-ROI' + '.png', 'png')

    print("Program finished!")


if __name__ == '__main__':
    dcms_rt_path = r"C:\Users\tingyi\Desktop\work\S729\data"
    out_path = r'C:\Users\tingyi\Desktop\work\S729\MAX_ROI_WITH_RESIZED_FIEXD_SIZE_IMG'
