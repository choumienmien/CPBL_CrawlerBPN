# CPBL_CrawlerBPN
 
## 說明
利用 Pytorch實作 Machine Learning 演算法 - 倒傳遞神經網路（BPN），並將模型的預測，運用在運動彩券中，進而推敲出下一場比賽的輸贏狀態

## 資料集（Dataset）
#### 利用API的方式，使用 [中華職業棒球大聯盟](https://www.cpbl.com.tw/)之數據

## 網絡說明

##### 三層網絡架構圖
<div  align="center">  
<img src="https://user-images.githubusercontent.com/37107594/212226808-891c7a72-cc0e-43f2-a202-615dea3e489c.png" alt="Cover" width="50%" />
</div>

#### 網路架構（Neural Network Architecture）
- 使用PyTorch框架創建一個帶有3層全連接網絡模型  
- 啟動函數(activation function )：非線性Mish函數以及Softmax演算法
- 損失函數:交叉熵(CrossEntropyLoss)
- 最佳化器(Optimizer)：Adam()
- 學習率衰減：ReduceLROnPlateau 

#### 驗證模型(Cross-Validation)
使用TimeSeriesSplit方法進行時序的劃分
<div  align="center">  
<img src="https://user-images.githubusercontent.com/37107594/212228383-97a22e80-394f-4f29-bebd-7223f5fbfa5a.png" alt="Cover" width="50%" />
</div>

- 資料分為「訓練階段(Train Phase)」以及「測試階段(Test Phase)」，為260筆與17筆
- 透過TimeSeriesSplit將訓練階段的數據，分為訓練集(Training Set)與驗證集(Valid Set)
- 採取時序性的方式將訓練資料劃分成4個fold


#### Optuna架構設定
1. 設定每個超參數搜尋範圍
   - dropout1(丟棄率1):	[0,1]
   - dropout2(丟棄率2):	[0,1]
   - learning_rate(學習率):[1e-5, 1e-1]
   - epochs(循環次數):[50,300]
   - unit1(隱藏層神經元個數1):[1,160]
   - Unit2(隱藏層神經元個數2):[1,160]

2. 以準確率及損失值之驗證分數進行探索
3. n_trials為[50,1500]，以每50為一區間，共計30個
4. 以Fold4之驗證集(52筆資料)與劃分於訓練集中的2021年數據(26筆)之準確率，進行最終最適超參數解的尋找


## 輸出結果
#### Optuna超參數最適解
<div  align="center">  
<img src="https://user-images.githubusercontent.com/37107594/212233262-045765ed-bbba-47c9-bf9b-5d77bab2eeb7.png" alt="Cover" />
</div>

- 隱藏層中神經元數目：第一層為146個神經元、第二層為28個神經元
- 丟棄率：第一層丟棄率0.07704681063017517、第二層丟棄率0.011019629397251096
- 循環次數：289
- 訓練的學習率：0.02551773884251893

##### 預測結果
- 召回率：100%
- 精確率:70%
- 準確率:82.3529%。
<div  align="center">  
<img src="https://user-images.githubusercontent.com/37107594/212234412-d0655bae-5ed2-40c1-87d5-001551c6633a.png" alt="Cover" width="50%"/>
</div>

## 貢獻

1. 爬取中華職棒網頁中的數據資料
2. 變數中加入進階數據，而非僅使用直觀數據
3. 考量數據含有時間資訊的變數
4. 透過Python中Pytorch框架建立模型(非使用統計軟體)
