# -*- coding: utf-8 -*-
import os
import shutil
from PIL import Image, ImageDraw
import glob
import sys, os, urllib.request, tarfile, cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data
from typing import Optional, List
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import random
from tqdm.notebook import tqdm
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import metrics
import timm
from timm.models.efficientnet import EfficientNet
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.spatial import distance
import csv
import re
import random
import time
import gc
import tracemalloc
import copy


####### input condition ############
save_RESIZE = 75 # imgサイズをマスク個数で割り切れる数値に
to_color = True
mask = True

level = 2
#model_name = 'tf_efficientnet_b6_ns'
#RESIZE = 528

model_name = 'tf_efficientnet_b7_ns'
RESIZE = 600

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device) #GPUが使用できているかの確認

model = timm.create_model(model_name, pretrained=True)
model.eval()
model.to(device)
################################

'''
フォルダの設定
    メインのフォルダ名 -> ユーザの名のフォルダ
                                -> 学習と検証データを含めたフォルダ名
                                -> train / validationの、2フォルダで以下共通
                                -> label_0 / label_1の、2フォルダで以下共通
                                -> それぞれの画像ファイル(png形式)

    ここのブロックで、上記のファイル名をそれぞれ入力
'''
main_folder_name = "Hazumi1911/png" #@param {type:"string"}
main_folder_dir = "./{}".format(main_folder_name)

# ラベルごとの画像を格納したフォルダをregular, irregularとして指定
# regularとして扱う画像を格納したフォルダ名
regular_label = "label_1" #@param {type:"string"}

# irregularとして扱う画像を格納したフォルダ名
irregular_label = "label_0" #@param {type:"string"}

# 使用する特徴量のフォルダを指定
feature_folder_name = "frame_sub" #@param {type:"string"}
seed_idx = 1 #@param {type:"integer"}

train_test_split_folder_name = "splited_test_{}_seed_{}".format(irregular_label, str(seed_idx))
train_path = train_test_split_folder_name + "/train"
test_path = train_test_split_folder_name + "/validation"

# 結果を記録するフォルダの作成
result_folder_name = "75_5" #@param {type:"string"}
result_path = "{}/result/{}/".format(main_folder_dir, result_folder_name)
try:
    result_folder = "{}/result".format(main_folder_dir)
    os.mkdir(result_folder)
except FileExistsError:
    print(result_folder + "は作成済み")
try:
    os.mkdir(result_path)
except FileExistsError:
    print(result_path + "は作成済み")
try:
    result_dir = os.path.join(result_path, train_test_split_folder_name)
    os.mkdir(result_dir)
except FileExistsError:
    print(result_dir + "は作成済み")
    

'''マスクパターンの生成
Input:
    n_maskpatterns: マスクパターンの数 (e.g. 1000)
    n_masks: マスク候補の位置の数

Returns:
    mask_datasheet: 2d numpy array, num_maskpattens * num_masks
'''

if mask == True:
    # 縦軸と横軸で情報が異なる場合(Skeleton(松藤データ))
    n_mask_in_a_col =  75#@param {type:"integer"}
    n_mask_in_a_row_time = 5#@param {type:"integer"}
    n_masks = n_mask_in_a_row_time * n_mask_in_a_col#n_mask_in_a_col**2

    masking_without_a_pixel = True #@param {type:"string"}
    identity_matrix = True #@param {type:"string"}


    if identity_matrix == True:
        # マスクをかける箇所は「0」
        # マスク以外の箇所を1つ指定
        if masking_without_a_pixel == True:
            mask_datasheet = np.eye(n_masks)
            mask_datasheet = np.insert(mask_datasheet, 0, 1, axis=0)
            n_maskpatterns = n_masks + 1

        # マスクの箇所を1つ指定
        else:
            mask_datasheet = np.eye(n_masks)
            mask_datasheet = abs(mask_datasheet - 1)
            mask_datasheet = np.insert(mask_datasheet, 0, 1, axis=0)
            n_maskpatterns = n_masks + 1
    else:
        np.random.seed(seed=32)
        mask_datasheet = np.random.randint(0, 2, (n_maskpatterns * n_masks)).reshape((n_maskpatterns, n_masks))
        mask_datasheet[0,:] = 1

    print("array_shape: ", mask_datasheet.shape)
    print("array: ", mask_datasheet)
else:
    n_maskpatterns = 1



if mask == True:
    ## マスクパターン(0, 1)の作成
    segments = []

    # マスクなしversionの作成(全てのpixelがオン / マスクがオフ:1が全て)
    #temp = np.ones((save_RESIZE, save_RESIZE, 3), dtype='int')
    temp = np.ones((save_RESIZE, save_RESIZE, 3), dtype='int')
    segments.append(temp)

    # マスクのサイズ
    feature_scale = int(save_RESIZE/n_mask_in_a_col) # 特徴量ごとに処理(列を指定)
    mask_scale = int(save_RESIZE/n_mask_in_a_row_time) # マスク/ピクセルの大きさ(行を指定)

    # マスクありversionを作成
    for col in range(n_mask_in_a_col):
        for row in range(n_mask_in_a_row_time):
            print("col: ",col)
            print("row: ",row)
            # マスクじゃないpixel位置(一箇所)とそれ以外はマスク箇所(1が一つ、0が他全て)
            if masking_without_a_pixel == True:
                # temp = np.zeros((save_RESIZE, save_RESIZE, 3), dtype='int')
                temp = np.zeros((save_RESIZE, save_RESIZE, 3), dtype='int')
                temp[row*mask_scale:(row+1)*mask_scale, col*feature_scale:(col+1)*feature_scale, :] = 1
                # numpy [行,列]
            else:
                # マスク位置(一箇所)とそれ以外はpixel(0が一つ、1が他全て)
                # temp = np.ones((save_RESIZE, save_RESIZE, 3), dtype='int')
                temp = np.ones((save_RESIZE, save_RESIZE, 3), dtype='int')
                temp[row*mask_scale:(row+1)*mask_scale, col*feature_scale:(col+1)*feature_scale, :] = 0
            temp = temp.astype(bool)
            segments.append(temp)
            

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

def MVTechAD(download_dir, path):
    target_path = "data/"

    if not os.path.exists(download_dir):
        os.mkdir(download_dir)

    # download file
    def _progress(count, block_size, total_size):
        sys.stdout.write('\rDownloading %s %.2f%%' % (source_path,
            float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

    source_path = path
    dest_path = os.path.join(download_dir, "data.tar.xz")
    urllib.request.urlretrieve(source_path, filename=dest_path, reporthook=_progress)
    # untar
    with tarfile.open(dest_path, "r:xz") as tar:
        tar.extractall(target_path)

class ImageTransform():
    def __init__(self, resize=RESIZE):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.Resize(resize),  # リサイズ
                #transforms.RandomCrop(224),
                #transforms.RandomRotation(20, fill=200),
                #transforms.RandomHorizontalFlip(),  # データオーギュメンテーション
                transforms.ToTensor(),  # テンソルに変換
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 0-1 → 標準化
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),  # リサイズ
                #transforms.CenterCrop(224),  # 画像中央をresize×resizeで切り取り
                transforms.ToTensor(),  # テンソルに変換
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 0-1 → 標準化
            ])
        }

    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)


def fig_show(img):
    # 2. 元の画像の表示
    plt.imshow(img)
    plt.title("Original")
    plt.show()

    transform = ImageTransform()
    img_transformed = transform(img, phase="train")  # torch.Size([3, 224, 224])

    # (色、高さ、幅)を (高さ、幅、色)に変換し、0-1に値を制限して表示
    plt.subplot(1,2,1)
    img_transformed = img_transformed.numpy().transpose((1, 2, 0))
    img_transformed = np.clip(img_transformed, 0, 1)
    plt.imshow(img_transformed*255)
    plt.title("Train")
    plt.show()

set_seed(1213)

def extract_features(inputs: torch.Tensor,
                     model: EfficientNet,
                     level):
    features = dict()
    # extract stem features as level 1
    x = model.conv_stem(inputs)
    x = model.bn1(x)
    x = model.act1(x)
    features['level_1'] = F.adaptive_avg_pool2d(x, 1)
    # extract blocks features as level 2~8
    for i, block_layer in enumerate(model.blocks):
        x = block_layer(x)
        features[f'level_{i+2}'] = F.adaptive_avg_pool2d(x, 1)
    # extract top features as level
    x = model.conv_head(x)
    x = model.bn2(x)
    x = model.act2(x)
    features['level_9'] = F.adaptive_avg_pool2d(x, 1)
    return features['level_{}'.format(str(level))]

# 複数枚(Dataloader, Dataset格納)からの特徴抽出
def get_mean_cov(loader):
    feat = []

    for inputs in loader:
        inputs = inputs.to(device)
        # levelは1~9のint, featuresは上述のextract_features()結果
        feat_list = extract_features(inputs, model, level)
        feat_list = feat_list.cpu().detach().numpy()
        # print(feat_list.shape)
        for i in range(len(feat_list)):
            feat.append(feat_list[i].reshape(-1))

    feat = np.array(feat)

    mean = np.mean(feat, axis=0)
    cov = np.cov(feat.T)

    return feat, mean, cov

# --------異常スコアの算出---------------------
def get_score(feat, mean, cov):
    result = []
    # 分散共分散行列の逆行列を計算
    cov_i = np.linalg.pinv(cov)

    for i in range(len(feat)):
        result.append(distance.mahalanobis(feat[i], mean, cov_i))
    return result, cov_i

# --------混合行列の算出---------------------
def get_evaluation(normal_score, anomaly_score, y_true):
    # 混合行列を算出
    # y_pred: np.hstack((normal_score, anomaly_score))
    cm = confusion_matrix(y_true, np.hstack((normal_score, anomaly_score)))
    accuracy = accuracy_score(y_true, np.hstack((normal_score, anomaly_score)))

    return cm, accuracy

def get_auc(Z1, Z2, save_name):
    mahalanobis_path = '{}_mahalanobis.png'.format(save_name)
    fig = plt.figure()
    plt.title("Mahalanobis distance")
    plt.plot(Z1, label="normal")
    plt.plot(Z2, label="anomaly")
    plt.legend()
    plt.savefig(mahalanobis_path)
    plt.close(fig)
    #plt.show()

    ## 正解ラベルの作成
    # 全て正常(0)として作成し、anormal_scoreに該当する部分を異常(1)に置換
    y_true = np.zeros(len(Z1)+len(Z2))
    y_true[len(Z1):] = 1#0:正常、1：異常

    # FPR, TPR(, しきい値) を算出
    #fpr, tpr, thres = metrics.roc_curve(y_true, y_pred) # hstack: 水平方向に連結
    fpr, tpr, _ = metrics.roc_curve(y_true, np.hstack((Z1, Z2)))

    # AUC
    auc = metrics.auc(fpr, tpr)

    return fpr, tpr, auc

def plot_roc(Z1, Z2, save_name, plot_name):
    fpr, tpr, auc1 = get_auc(Z1, Z2, save_name)
    roc_path = '{}_ROC.png'.format(save_name)
    
    fig = plt.figure()
    plt.plot(fpr, tpr, label=plot_name + '(AUC = %.3f)'%(auc1))
    plt.legend()
    plt.title(plot_name + '(ROC)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    #plt.savefig(roc_path)
    plt.savefig(roc_path)
    #plt.show()
    plt.close(fig)

    return auc1

        
        
def generate_maskImage(img_array, mask_datasheet, masking_idx):
    # 縦・横を指定した数値に分割し、マスク部分を０、それ以外を１とした行列作成
    # 元画像との乗算に使用
    ## 初回で計算済み
    #segments = ImageSegmentation(img_array)

    # 元画像のコピー(マスク処理用)
    temp = copy.deepcopy(img_array)
    
    # masking_idx番号に相当するマスクパターン
    #row = mask_datasheet[masking_idx]
    
    # マスクをかける箇所(0)のマスク番号（列番号）を抽出
    #zeros = np.where(row == 0)[0]
    
    # マスクをかける箇所(0)を元画像のコピー(temp)に乗算
    # 0 * 画素値の計算で、画素値を削除（マスク）する。
    #for z in zeros:
    #    temp = segments[z].astype(int) * temp

    ## マスク番号にあったマスクパターンのみを乗算
    temp = segments[masking_idx].astype(int) * temp
    
    return temp

# マスクを施すデータセット作成関数
class MaskDataset(data.Dataset):
    def __init__(self,
                 train_list: List[List[str]],
                 to_color="False",
                 masking_idx=0):
                 
        self.file_list = train_list
        self.to_color = to_color
        self.transform = ImageTransform()
        self.masking_idx = masking_idx

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int):
        # 訓練、テスト画像のパスを格納したリスト
        img_path = self.file_list[idx]
        if self.to_color:
            img = Image.open(img_path).convert("RGB")  # [高さ][幅][色RGB]
        else:
            img = Image.open(img_path)

        if mask == True:
            img= img.resize((save_RESIZE, save_RESIZE))
            # PIL Image -> np.array
            img_array = np.array(img)
            
            # getitem関数による画像idx・masking_idxに対応するマスク画像arrayを算出
            masked_img_array = generate_maskImage(img_array, mask_datasheet, self.masking_idx)
            
            # np.array -> PIL Image
            img = Image.fromarray(masked_img_array.astype(np.uint8))

        # Imageをリサイズ / ToTensor / RGBの正規化
        img = self.transform(img, "train")
        
        return img
        

'''
EfficientNet
'''
# dumpfileに1911F7001.csvなどが含まれていなかったので、リストから削除
user_list = ["1911F2001", "1911F2002", "1911F3001", "1911F3002", "1911F3003",
             "1911F4001", "1911F4002", "1911F4003", "1911F5001", "1911F5002",
             "1911F6001", "1911F6002", "1911F6003", "1911F7002", "1911M2001",
             "1911M2002", "1911M2003", "1911M4001", "1911M4002", "1911M5001",
             "1911M5002", "1911M6001", "1911M6002", "1911M6003", "1911M7001", "1911M7002"]

auc_list = []
for file_idx in range(0, len(user_list)):
    '''
    該当ユーザの訓練、テスト画像のパスを取得
    '''
    start_user = time.time()
    target_folder_name = main_folder_dir + "/" + user_list[file_idx] + "/experiment"
    print(target_folder_name)
    
    # Regularと指定したラベルに従って、学習データと検証データの画像が格納されたフォルダへのパスを指定
    train_img_file_path = "{}/{}/{}".format(target_folder_name, train_path, regular_label)
    test_img_file_path = "{}/{}".format(target_folder_name, test_path)
    train_list = glob.glob(train_img_file_path + "/**.png")
    test_list = glob.glob(test_img_file_path + "/**/**.png")

    normal_list = glob.glob("{}/{}/**.png".format(test_img_file_path, regular_label))
    anomaly_list = glob.glob("{}/{}/**.png".format(test_img_file_path, irregular_label))

    print("学習データ(正常データのみ) : ",len(train_list))
    print("テストデータ(正常、異常のmix) : ",len(test_list))
    print("テストデータ(正常) : ",len(normal_list))
    print("テストデータ(異常) : ",len(anomaly_list))

    # DataFrame形式で画像ファイル名と異常スコアを保存
    normal_df = pd.DataFrame({"normal_file_list": normal_list})
    anomaly_df = pd.DataFrame({"anomaly_file_list": anomaly_list})

    '''
    EfficientNetを用いた全マスク画像の特徴抽出
    正解分布作成と異常データのマハラノビス距離の計算
    '''
    for masking_idx in range(0, n_maskpatterns):
        start_mask = time.time()
        
        # マスクを施したデータセットを作成
        train_dataset = MaskDataset(
                                    train_list,
                                    to_color=to_color,
                                    masking_idx=masking_idx
                                    )
        normal_dataset = MaskDataset(
                                    normal_list,
                                    to_color=to_color,
                                    masking_idx=masking_idx
                                    )
        anomaly_dataset = MaskDataset(
                                    anomaly_list,
                                    to_color=to_color,
                                    masking_idx=masking_idx
                                    )

        # 各データセットをPytorchの学習に用いるデータ形式へ変更
        # 今回はバッチサイズは1にしているため、データ形式を変換しているのみ
        train_loader = data.DataLoader(
                                        train_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=2,
                                        pin_memory=True,
                                        drop_last=True
                                        )

        normal_loader = data.DataLoader(
                                        normal_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=2,
                                        # pin_memory=True,
                                        drop_last=True
                                        )

        anomaly_loader = data.DataLoader(
                                        anomaly_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=2,
                                        # pin_memory=True,
                                        drop_last=True
                                        )

        # 複数画像のEfficientNetの中間層での特徴を抽出
        # 訓練画像での特徴量の平均、分散を計算
        train_feat, mean, cov = get_mean_cov(train_loader)
        normal_feat, _, _ = get_mean_cov(normal_loader)
        anomaly_feat, _, _ = get_mean_cov(anomaly_loader)

        # 「訓練画像の平均と分散」とテストデータを比較し、異常スコアを算出
        normal_score, cov_i = get_score(normal_feat, mean, cov)
        anomaly_score, _ = get_score(anomaly_feat, mean, cov)
        # Numpyへ変換
        normal_score = np.array(normal_score)
        anomaly_score = np.array(anomaly_score)

        # 各異常スコアを各マスキング状況でDataFrame形式で保存
        normal_score_df = pd.DataFrame({masking_idx: normal_score})
        anomaly_score_df = pd.DataFrame({masking_idx: anomaly_score})
        
        # オリジナル画像での異常スコアを用いてROC曲線を描画
        ## google driveに保存
        save_path = "{}/{}/{}".format(result_path, train_test_split_folder_name, user_list[file_idx])
        plot_path = user_list[file_idx]
        auc = plot_roc(normal_score, anomaly_score, save_path, plot_path)#train(フォルダ名)はグラフ化の際に使用
        auc_list.append(auc)

        # DataFrameの更新
        normal_df = pd.concat([normal_df, normal_score_df], axis=1)
        anomaly_df = pd.concat([anomaly_df, anomaly_score_df], axis=1)
        print("user: ", user_list[file_idx], ", mask_id: ", masking_idx)

        # ユーザごとの全てのプロセス終了までに経過した時間
        elapsed_time_mask = time.time() - start_mask
        print ("mask_process_elapsed_time:{0}".format(elapsed_time_mask) + "[sec]")

    # DataFrame -> csv
    csv_name_normal = "{}/{}/{}_{}_normal_df.csv".format(result_path, train_test_split_folder_name, user_list[file_idx], regular_label)
    normal_df.to_csv(csv_name_normal)
    
    csv_name_anomaly = "{}/{}/{}_{}_anomaly_df.csv".format(result_path, train_test_split_folder_name, user_list[file_idx], regular_label)
    anomaly_df.to_csv(csv_name_anomaly)

    # ユーザごとの終了までに経過した時間
    elapsed_time_user = time.time() - start_user
    print ("user_process_elapsed_time:{0}".format(elapsed_time_user) + "[sec]")

auc_df = pd.DataFrame(auc_list)
auc_df.index = user_list
csv_save_path = "{}/{}/auc.csv".format(result_path, train_test_split_folder_name)
auc_df.to_csv(csv_save_path)
