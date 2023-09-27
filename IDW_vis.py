import numpy as np 
import matplotlib.pyplot as plt
from cnmaps import get_adm_maps
import pandas as pd
import cv2
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial import cKDTree as KDTree

class Invdisttree:

    def __init__(self, X, z, leafsize=5, stat=0):
        assert len(X) == len(z), "len(X) %d != len(z) %d" % (len(X), len(z))
        self.tree = KDTree(X, leafsize=leafsize)  # build the tree
        self.z = z
        self.stat = stat
        self.wn = 0
        self.wsum = None

    def __call__(self, q, nnear=6, eps=0.0, p=1, weights=None, max_dist=None):
        q = np.asarray(q)
        qdim = q.ndim
        if qdim == 1:
            q = np.array([q])
        if self.wsum is None:
            self.wsum = np.zeros(nnear)
        self.distances, self.ix = self.tree.query(q, k=nnear, eps=eps)
        
        interpol = np.zeros((len(self.distances),) + np.shape(self.z[0]))

        jinterpol = 0
        for dist, ix in zip(self.distances, self.ix):
            if max_dist is not None:  # 如果设置了最大距离
                valid_indices = dist < max_dist  # 找出距离小于最大距离的邻居
                dist = dist[valid_indices]  # 只考虑这些邻居的距离
                ix = ix[valid_indices]  # 只考虑这些邻居的索引
            if len(dist) == 0:  # 如果没有满足条件的邻居，跳过这个点
                continue
            elif len(dist) == 1:  # 如果只有一个满足条件的邻居，直接使用该邻居的值
                wz = self.z[ix]
            elif dist[0] < 1e-10:  # 如果最近的邻居的距离非常小，直接使用最近邻的值
                import pdb;pdb.set_trace()
                wz = self.z[ix[0]]
            else:  # 否则，使用反距离权重插值
                w = 1 / dist**p
                if weights is not None:
                    w *= weights[ix]  # >= 0
                w /= np.sum(w)
                wz = np.dot(w, self.z[ix])
                if self.stat:
                    self.wn += 1
                    self.wsum += w
            interpol[jinterpol] = wz
            jinterpol += 1
        return interpol if qdim > 1 else interpol[0]


def vis_vis(input_path):
    # Paras
    dpi=0.025                       # 空间分辨率
    lon_l = 135.1873203714432       # 经度
    lon_s = 73.39911225249426
    lat_l = 53.66164651851026       # 纬度
    lat_s = 3.737891241334588
    lon_inter = int((lon_l-lon_s)/dpi)+2
    lat_inter = int((lat_l-lat_s)/dpi)-3
    # Fog不同等级的可视化对应的颜色:0-5
    colors=[(255,255,255),(231,255,197),(254,255,99),(205,207,1),(156,154,0),(189,93,0)]
    
    # lon = np.linspace(lon_s, lon_l, lon_inter)
    # lat = np.linspace(lat_s, lat_l, lat_inter)
    lon = np.linspace(lon_s, lon_l, lon_inter)
    lat = np.linspace(lat_l, lat_s, lat_inter)
    lons, lats = np.meshgrid(lon, lat)
    
    # 利用cnmaps读取中国国界图
    if(not os.path.exists("China_all.npy")):
        china = get_adm_maps(level="国", record="first", only_polygon=True, wgs84=True)
        mask = china.make_mask_array(lons, lats)
        print(mask.shape)
        # plt.imshow(mask, cmap='binary', origin='lower')
        # np.save("China_all.npy", mask)
    else:
        print("load data\n")
        mask = np.load("China_all.npy")
    
    data = pd.read_csv(input_path,sep='\s+',header=0,skiprows=3,encoding="gbk",usecols=[0,1,2,3,17,18],
                    names=['station','lon','lat','high','vis','code'])
    print(data.shape)
    fog_data = data
    
    # 将原来的能见度数据转化成0-5等级
    fog_vis = np.array(fog_data['vis'].to_list())
    fog_vis[fog_vis==0] = 0.01    # 先将0置为0.01
    fog_vis[fog_vis==9999] = 0    # 缺失值9999设置为0
    fog_vis[fog_vis<0] = 0
    fog_vis = fog_vis*1000
    fog_vis[(fog_vis>0)*(fog_vis<50)]       = 5
    fog_vis[(fog_vis>=50)*(fog_vis<200)]    = 4
    fog_vis[(fog_vis>=200)*(fog_vis<500)]   = 3
    fog_vis[(fog_vis>=500)*(fog_vis<1000)]  = 2
    fog_vis[(fog_vis>=1000)*(fog_vis<10000)]= 1
    fog_vis[(fog_vis>=10000)]               = 0
    fog_data['vis'] = fog_vis.tolist()
    
    longitude = fog_data['lon'].tolist()
    latitude = fog_data['lat'].tolist()
    
    longitude = list(map(float,longitude))
    latitude = list(map(float,latitude))
    
    fog_value = fog_data['vis'].tolist()
    fog_value = np.array(fog_value,dtype=np.float64)
    
    X = np.array([[x,y] for x, y in zip(longitude,latitude)])
    
    invdisttree = Invdisttree(X, fog_value, leafsize=10, stat=0)
    
    Data_res = []  # 反距离权重插值后 得到的结果 list存储的每一行插值结果
    for i in tqdm(range(0,lats.shape[0])):
        X_new = np.array([[x, y] for x, y in zip(lons[i], lats[i])])
        interpol = invdisttree(X_new, nnear=6, eps=0.1, p=2, max_dist=25)
        Data_res.append(interpol)
    
    # import pdb;pdb.set_trace()
    np_data = np.array(Data_res)
    np_data[mask==True] = 0
    res=np.zeros((mask.shape[0],mask.shape[1],3),dtype=np.uint8)
    
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j]!=True:
                level = round(np_data[i][j])
                res[i][j] = colors[level]
            else:
                res[i][j] = (236,236,236)
    
    return res


path='/Users/kaka/Desktop/Inter/2022121012.000'
fog_data=vis_vis(path)
cv2.imwrite('test.jpg',fog_data)