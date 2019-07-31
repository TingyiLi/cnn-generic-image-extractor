from radiomics import featureextractor
import SimpleITK as sitk
import six
import os
import pandas as pd


def extract(setting_path, sample_nrrd_path, smaple_nrrd_seg_path, save_path):
    image = sitk.ReadImage(sample_nrrd_path)
    mask = sitk.ReadImage(smaple_nrrd_seg_path)

    extractor = featureextractor.RadiomicsFeaturesExtractor(setting_path)
    # 计算特征
    result = extractor.execute(image, mask)
    RESULT = {}  # 保存结果
    RESULT['Patient ID'] = ['001']
    for key, val in six.iteritems(result):
        # print("\t%s: %s" % (key, val))
        RESULT[key] = [str(val)]
    #
    # 字典中的key值即为csv中列名
    dataframe = pd.DataFrame(RESULT)
    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv(os.path.join(save_path, "result.csv"), index=False, sep=',')


if __name__ == "__main__":
    sample_nrrd_path = ""
    sample_nrrd_seg_path = ""
    setting_path = ""
    save_path = ""
    extract(setting_path, sample_nrrd_path, sample_nrrd_seg_path, save_path)
