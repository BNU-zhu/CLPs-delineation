'''
This code is used to process the whole image and boundary/region samples into 256*256 sample tiles 
and calculate the distance samples based on the region samples
'''
import os
import cv2 as cv
import numpy as np
from PIL import Image
from osgeo import gdal
#  读取tif数据集
def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    return dataset
    
#  保存tif文件函数
def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
    #创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if(dataset!= None):
        dataset.SetGeoTransform(im_geotrans) #写入仿射变换参数
        dataset.SetProjection(im_proj) #写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset

def read_img(filename):
    dataset=gdal.Open(filename)

    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize

    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    im_data = dataset.ReadAsArray(0,0,im_width,im_height)

    del dataset
    return im_proj, im_geotrans, im_width, im_height, im_data


def write_img(filename, im_proj, im_geotrans, im_data):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1,im_data.shape

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    dataset.SetGeoTransform(im_geotrans)
    dataset.SetProjection(im_proj)

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i+1).WriteArray(im_data[i])

    del dataset  

'''
滑动窗口裁剪函数
TifPath 影像路径
SavePath 裁剪后保存目录
CropSize 裁剪尺寸
RepetitionRate 重复率
'''
def TifCrop(TifPath, SavePath, CropSize, RepetitionRate):
    dataset_img = readTif(TifPath)
    width = dataset_img.RasterXSize
    height = dataset_img.RasterYSize
    proj = dataset_img.GetProjection()
    geotrans = dataset_img.GetGeoTransform()
    img = dataset_img.ReadAsArray(0, 0, width, height)#获取数据
    
    #  获取当前文件夹的文件个数len,并以len+1命名即将裁剪得到的图像
    new_name = len(os.listdir(SavePath)) + 1
    #  裁剪图片,重复率为RepetitionRate
    
    for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
        for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
            #  如果图像是单波段
            if(len(img.shape) == 2):
                cropped = img[int(i * CropSize * (1 - RepetitionRate)) : int(i * CropSize * (1 - RepetitionRate)) + CropSize, 
                              int(j * CropSize * (1 - RepetitionRate)) : int(j * CropSize * (1 - RepetitionRate)) + CropSize]
            #  如果图像是多波段
            else:
                cropped = img[:,
                              int(i * CropSize * (1 - RepetitionRate)) : int(i * CropSize * (1 - RepetitionRate)) + CropSize, 
                              int(j * CropSize * (1 - RepetitionRate)) : int(j * CropSize * (1 - RepetitionRate)) + CropSize]
            #  写图像
            #im = Image.fromarray(cropped.transpose(1,2,0))
            #im.save(SavePath + "/%d.png"%new_name)
            #writepng(cropped, SavePath + "/%d.png"%new_name)
            writeTiff(cropped, geotrans, proj, SavePath + "/%d.tif"%new_name)
            #  文件名 + 1
            new_name = new_name + 1
    #  向前裁剪最后一列
    for i in range(int((height-CropSize*RepetitionRate)/(CropSize*(1-RepetitionRate)))):
        if(len(img.shape) == 2):
            cropped = img[int(i * CropSize * (1 - RepetitionRate)) : int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                          (width - CropSize) : width]
        else:
            cropped = img[:,
                          int(i * CropSize * (1 - RepetitionRate)) : int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                          (width - CropSize) : width]
        #  写图像
        #writepng(cropped, SavePath + "/%d.png"%new_name)
        #im = Image.fromarray(cropped.transpose(1,2,0))
        #im.save(SavePath + "/%d.png"%new_name)
        writeTiff(cropped, geotrans, proj, SavePath + "/%d.tif"%new_name)
        new_name = new_name + 1
    #  向前裁剪最后一行
    for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
        if(len(img.shape) == 2):
            cropped = img[(height - CropSize) : height,
                          int(j * CropSize * (1 - RepetitionRate)) : int(j * CropSize * (1 - RepetitionRate)) + CropSize]
        else:
            cropped = img[:,
                          (height - CropSize) : height,
                          int(j * CropSize * (1 - RepetitionRate)) : int(j * CropSize * (1 - RepetitionRate)) + CropSize]
        #writepng(cropped, SavePath + "/%d.png"%new_name)
        #im = Image.fromarray(cropped.transpose(1,2,0))
        #im.save(SavePath + "/%d.png"%new_name)
        writeTiff(cropped, geotrans, proj, SavePath + "/%d.tif"%new_name)
        #  文件名 + 1
        new_name = new_name + 1
    #  裁剪右下角
    if(len(img.shape) == 2):
        cropped = img[(height - CropSize) : height,
                      (width - CropSize) : width]
    else:
        cropped = img[:,
                      (height - CropSize) : height,
                      (width - CropSize) : width]
    #writepng(cropped, SavePath + "/%d.png"%new_name)
    #im = Image.fromarray(cropped.transpose(1,2,0))
    #im.save(SavePath + "/%d.png"%new_name)
    writeTiff(cropped, geotrans, proj, SavePath + "/%d.tif"%new_name)
    new_name = new_name + 1

imageRoot = r"./ImageDataset/data/image/P1.tif"
regionRoot = r"./ImageDataset/data/region/P1.tif"
boundaryRoot = r"./ImageDataset/data/boundary/P1.tif"

crop_imageRoot = r"./ImageDataset/train/image"
crop_regionRoot = r"./ImageDataset/train/region"
crop_boundaryRoot = r"./ImageDataset/train/boundary"
crop_distRoot = r"./ImageDataset/train/distance"
    
#  将影像1裁剪为重复率为0.2的256×256的数据集
TifCrop(imageRoot,
        crop_imageRoot, 256, 0.2)
TifCrop(regionRoot,
        crop_regionRoot, 256, 0.2)
TifCrop(boundaryRoot,
        crop_boundaryRoot, 256, 0.2)


np.seterr(divide='ignore',invalid='ignore')
for imgPath in os.listdir(crop_regionRoot):
    input_path = os.path.join(crop_regionRoot, imgPath)
    distOutPath = os.path.join(crop_distRoot, imgPath)
    im_proj, im_geotrans, im_width, im_height, im_data = read_img(input_path)
    result = cv.distanceTransform(src=im_data, distanceType=cv.DIST_L2, maskSize=3)
    min_value = np.min(result)
    max_value = np.max(result)
    scaled_image = ((result - min_value) / (max_value - min_value)) * 255
    result = scaled_image.astype(np.uint8)
    write_img(distOutPath, im_proj, im_geotrans, result)
