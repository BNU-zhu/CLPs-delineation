import torch
from torchvision import transforms
from model import Our_Model
from osgeo import gdal
import numpy as np
import math
import cv2

    
# 读取tif数据集
def readTif(fileName, xoff = 0, yoff = 0, data_width = 0, data_height = 0):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    #  栅格矩阵的列数
    width = dataset.RasterXSize 
    #  栅格矩阵的行数
    height = dataset.RasterYSize 
    #  获取数据
    if(data_width == 0 and data_height == 0):
        data_width = width
        data_height = height
    data = dataset.ReadAsArray(xoff, yoff, data_width, data_height)

    return data

# 保存tif文件函数
def writeTiff(fileName, data, im_geotrans=(0,0,0,0,0,0), im_proj=""):
    if 'int8' in data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(data.shape) == 3:
        im_bands, im_height, im_width = data.shape
    elif len(data.shape) == 2:
        data = np.array([data])
        im_bands, im_height, im_width = data.shape

    #创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(fileName, int(im_width), int(im_height), int(im_bands), datatype)
    if(dataset!= None):
        dataset.SetGeoTransform(im_geotrans) #写入仿射变换参数
        dataset.SetProjection(im_proj) #写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i+1).WriteArray(data[i])
    del dataset

#  tif裁剪（tif像素数据，裁剪边长）
def TifCroppingArray(img, SideLength, Size):
    #  裁剪链表
    TifArrayReturn = []
    #  列上图像块数目
    ColumnNum = int((img.shape[0] - SideLength * 2) / (Size - SideLength * 2))
    #  行上图像块数目
    RowNum = int((img.shape[1] - SideLength * 2) / (Size - SideLength * 2))
    for i in range(ColumnNum):
        TifArray = []
        for j in range(RowNum):
            cropped = img[i * (Size - SideLength * 2) : i * (Size - SideLength * 2) + Size,
                          j * (Size - SideLength * 2) : j * (Size - SideLength * 2) + Size]
            TifArray.append(cropped)
        TifArrayReturn.append(TifArray)
    #  考虑到行列会有剩余的情况，向前裁剪一行和一列
    #  向前裁剪最后一列
    for i in range(ColumnNum):
        cropped = img[i * (Size - SideLength * 2) : i * (Size - SideLength * 2) + Size,
                      (img.shape[1] - Size) : img.shape[1]]
        TifArrayReturn[i].append(cropped)
    #  向前裁剪最后一行
    TifArray = []
    for j in range(RowNum):
        cropped = img[(img.shape[0] - Size) : img.shape[0],
                      j * (Size-SideLength*2) : j * (Size - SideLength * 2) + Size]
        TifArray.append(cropped)
    #  向前裁剪右下角
    cropped = img[(img.shape[0] - Size) : img.shape[0],
                  (img.shape[1] - Size) : img.shape[1]]
    TifArray.append(cropped)
    TifArrayReturn.append(TifArray)
    #  列上的剩余数
    ColumnOver = (img.shape[0] - SideLength * 2) % (Size - SideLength * 2) + SideLength
    #  行上的剩余数
    RowOver = (img.shape[1] - SideLength * 2) % (Size - SideLength * 2) + SideLength
    return TifArrayReturn, RowOver, ColumnOver

#  获得结果矩阵
def Result(shape, TifArray, npyfile, RepetitiveLength, RowOver, ColumnOver, Size):
    result = np.zeros(shape, np.uint8)
    #  j来标记行数
    j = 0  
    for i,img in enumerate(npyfile):
        #  最左侧一列特殊考虑，左边的边缘要拼接进去
        if(i % len(TifArray[0]) == 0):
            #  第一行的要再特殊考虑，上边的边缘要考虑进去
            if(j == 0):
                result[0 : Size - RepetitiveLength, 0 : Size-RepetitiveLength] = img[0 : Size - RepetitiveLength, 0 : Size - RepetitiveLength]
            #  最后一行的要再特殊考虑，下边的边缘要考虑进去
            elif(j == len(TifArray) - 1):
                #  原来错误的
                #result[shape[0] - ColumnOver : shape[0], 0 : Size - RepetitiveLength] = img[0 : ColumnOver, 0 : Size - RepetitiveLength]
                #  后来修改的
                result[shape[0] - ColumnOver - RepetitiveLength: shape[0], 0 : Size - RepetitiveLength] = img[Size - ColumnOver - RepetitiveLength : Size, 0 : Size - RepetitiveLength]
            else:
                result[j * (Size - 2 * RepetitiveLength) + RepetitiveLength : (j + 1) * (Size - 2 * RepetitiveLength) + RepetitiveLength,
                       0:Size-RepetitiveLength] = img[RepetitiveLength : Size - RepetitiveLength, 0 : Size - RepetitiveLength]   
        #  最右侧一列特殊考虑，右边的边缘要拼接进去
        elif(i % len(TifArray[0]) == len(TifArray[0]) - 1):
            #  第一行的要再特殊考虑，上边的边缘要考虑进去
            if(j == 0):
                result[0 : Size - RepetitiveLength, shape[1] - RowOver: shape[1]] = img[0 : Size - RepetitiveLength, Size -  RowOver: Size]
            #  最后一行的要再特殊考虑，下边的边缘要考虑进去
            elif(j == len(TifArray) - 1):
                result[shape[0] - ColumnOver : shape[0], shape[1] - RowOver : shape[1]] = img[Size - ColumnOver : Size, Size - RowOver : Size]
            else:
                result[j * (Size - 2 * RepetitiveLength) + RepetitiveLength : (j + 1) * (Size - 2 * RepetitiveLength) + RepetitiveLength,
                       shape[1] - RowOver : shape[1]] = img[RepetitiveLength : Size - RepetitiveLength, Size - RowOver : Size]   
            #  走完每一行的最右侧，行数+1
            j = j + 1
        #  不是最左侧也不是最右侧的情况
        else:
            #  第一行的要特殊考虑，上边的边缘要考虑进去
            if(j == 0):
                result[0 : Size - RepetitiveLength,
                       (i - j * len(TifArray[0])) * (Size - 2 * RepetitiveLength) + RepetitiveLength : (i - j * len(TifArray[0]) + 1) * (Size - 2 * RepetitiveLength) + RepetitiveLength
                       ] = img[0 : Size - RepetitiveLength, RepetitiveLength : Size - RepetitiveLength]         
            #  最后一行的要特殊考虑，下边的边缘要考虑进去
            if(j == len(TifArray) - 1):
                result[shape[0] - ColumnOver : shape[0],
                       (i - j * len(TifArray[0])) * (Size - 2 * RepetitiveLength) + RepetitiveLength : (i - j * len(TifArray[0]) + 1) * (Size - 2 * RepetitiveLength) + RepetitiveLength
                       ] = img[Size - ColumnOver : Size, RepetitiveLength : Size - RepetitiveLength]
            else:
                result[j * (Size - 2 * RepetitiveLength) + RepetitiveLength : (j + 1) * (Size - 2 * RepetitiveLength) + RepetitiveLength,
                       (i - j * len(TifArray[0])) * (Size - 2 * RepetitiveLength) + RepetitiveLength : (i - j * len(TifArray[0]) + 1) * (Size - 2 * RepetitiveLength) + RepetitiveLength,
                       ] = img[RepetitiveLength : Size - RepetitiveLength, RepetitiveLength : Size - RepetitiveLength]
    return result

area_perc = 0.5
TifPath = r"./shandong1.tif"
Size = 256
model_path = r"./44.pt"
        
ResultPath = r"./shandong1_region_86_1024.tif"
RepetitiveLength = int((1 - math.sqrt(area_perc)) * Size / 2)

big_image = cv2.imread(TifPath, cv2.IMREAD_UNCHANGED)
TifArray, RowOver, ColumnOver = TifCroppingArray(big_image, RepetitiveLength, Size)

data_transforms = transforms.Compose(
   [
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

   ]
)

model = Our_Model(num_classes=1)

# load model to DEVICE
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu' 
model.to(DEVICE)
checkpoint = torch.load(model_path)

# Remove unnecessary prefix 'module.' from state_dict
if 'module.' in list(checkpoint.keys())[0]:
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
else:
    state_dict = checkpoint
 
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()
predicts = []
for i in range(len(TifArray)):
    for j in range(len(TifArray[0])):
        image = TifArray[i][j]
        image = data_transforms(image)
        image = image.cuda()[None]
        pred = np.zeros((Size,Size))       
        with torch.no_grad():
            outputs1, outputs2, outputs3 = model(image)
        pred = outputs1.detach().cpu().numpy().squeeze()
        pred[pred>0] = 255
        pred[pred<=0] = 0
        #pred = pred * 255
        #print(pred)
        predicts.append((pred))

#保存结果predictspredicts
result_shape = (big_image.shape[0], big_image.shape[1])
result_data = Result(result_shape, TifArray, predicts, RepetitiveLength, RowOver, ColumnOver, Size)
writeTiff(ResultPath, result_data)
