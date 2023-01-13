# -*- coding: utf-8 -*-
'''
更改:
1.沿用V5


5-1 split分4份
v1:1-17(15)
v2:18-34(16-30)
v3:35-51(31-45)
v4:52-68(46-60)

5-2 split分5份
v1:1-17(61-75)
v2:18-34(76-90)
v3:35-51(91-105)
v4:52-68(106-120)
'''

import csv
import glob
import os.path
import re
import sys
import time
import plotly
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim

from sklearn.model_selection import TimeSeriesSplit
import optuna


class Mish(nn.Module):  # Mish激活函数
    def __init__(self):
        super().__init__()
        print("Mish activation loaded...")

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


class ThreelinearModel(nn.Module):
    def __init__(self, i, j, m, n):
        super().__init__()
        self.linear1 = nn.Linear(68, i)  # 定義全連接層
        self.dropout1 = nn.Dropout(m)
        self.mish1 = Mish()
        self.linear2 = nn.Linear(i, j)  # 定義全連接層
        self.dropout2 = nn.Dropout(n)
        self.mish2 = Mish()
        self.linear3 = nn.Linear(j, 3)  # 定義全連接層
        self.softmax = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss()  # 定义交叉熵函数

    def forward(self, x):  # 定义一个全连接网络(架設用兩個全連結層組成的網路模型)
        lin1_out = self.linear1(x)  # 將輸入資料傳入第一個全連接層
        drop = self.dropout1(lin1_out)
        out1 = self.mish1(drop)  # 對第一個連接層的結果進行Mish
        drop2 = self.dropout2(out1)
        out2 = self.mish2(self.linear2(drop2))  # 將網路資料進行傳入第二個連接層
        return self.softmax(self.linear3(out2))

    def getloss(self, x, y):  # 实现LogicNet类的损失值计算接口
        y_pred = self.forward(x)
        loss = self.criterion(y_pred, y)  # 计算损失值得交叉熵(預測結果與目標之間誤差的交叉熵)
        return loss


# 轉換原Excel資料
def TransformData(OriginData):
    Union = pd.read_csv(OriginData, delimiter=",")
    Union['IntPitcherHabit'] = Union['PitcherHabit'].astype(int)
    Union['IntBrother.PitcherHabit'] = Union['Brother.PitcherHabit'].astype(int)
    Union['IntVisitingHomeType'] = Union['VisitingHomeType'].astype(int)
    Union['IntWinLoss'] = Union['WinLoss'].astype(int)
    Union = pd.concat([Union,
                       pd.get_dummies(Union['DayNight'], prefix='DayNight'),
                       pd.get_dummies(Union['IntPitcherHabit'], prefix='PitcherHabit'),
                       pd.get_dummies(Union['IntBrother.PitcherHabit'], prefix='Brother.PitcherHabit'),
                       pd.get_dummies(Union['IntVisitingHomeType'], prefix='VisitingHomeType'),
                       pd.get_dummies(Union['IntWinLoss'], prefix='WinLoss')
                       ], axis=1)

    UnionDF = Union.drop(
        ['GameDate', 'MinusScore', 'IntPitcherHabit', 'PitcherHabit', 'IntBrother.PitcherHabit', 'Brother.PitcherHabit',
         'VisitingHomeType','IntVisitingHomeType', 'DayNight', 'WinLoss', 'IntWinLoss',
         'VisitingHomeType_2','DayNight_N'], axis=1)
    # print(UnionDF.columns)
    # exit()
    # print(UnionDF['VisitingHomeType_1'])
    UnionDF_Data = UnionDF.iloc[:260, :]
    UnionDF_Exam = UnionDF.iloc[260:, :]
    UnionDF_Valid = UnionDF.iloc[234:260, :]
    UnionDF_Valid2 = UnionDF.iloc[208:260, :]
    UnionDF.to_csv("Union.csv")
    UnionDF_Data.to_csv("UnionTrain.csv")
    UnionDF_Exam.to_csv("UnionTest.csv")  # 儲存資料
    UnionDF_Valid.to_csv("UnionValid.csv")
    UnionDF_Valid2.to_csv("UnionValid2.csv")


torch.manual_seed(0)  # 设置随机种子


class PytorchAdam:
    def __int__(self):
        self.data = []  # 存放數據(結果)的地方

    def path_num(self):
        # files = os.listdir("C:/Users/mandy chou/Documents/GitHub/BPN_CPBL/PTFile")  # 读入文件夹
        path = glob.glob(os.path.join("PTFile", "*"))
        UseNumber = []
        for i in path:
            directory, file_name = os.path.split(i)
            pattern = '(\d+)'
            num = int(re.findall(pattern, file_name)[-1])
            UseNumber.append(num)

        if len(UseNumber) == 0:
            maxnum = 0
        else:
            maxnum = max(UseNumber)  # 統計目前使用文件最大編號(會在創建新資料夾之前的筆數)

        self.current_number = maxnum
        self.folder = "C:/Users/mandy chou/Documents/GitHub/BPN_CPBL/PTFile/folder{}".format(self.current_number)

    def normalized(self, x_data, y_data):
        e = 1e-7  # 防止出现0
        for i in range(x_data.shape[1]):
            max_num = np.max(x_data[:, i])
            min_num = np.min(x_data[:, i])
            x_data[:, i] = (x_data[:, i] - min_num + e) / (max_num - min_num + e)
        y_data = (y_data - np.min(y_data) + e) / (np.max(y_data) - np.min(y_data) + e)
        return x_data, y_data

    def train(self, Filename, net, lr, epochs, hidden_1, hidden_2, dropout1, dropout2):
        Union = pd.read_csv(Filename, delimiter=",")
        # 取所有的行，和第一列之后的数据，因为第一列是标签，后面的是特征
        # 會取得標籤A_MinusTotalScore	A_DayNight	B_VisitingHomeType	Pre_League_ERA	Pre_Personal_IP.....
        # print(Union.columns)
        Y = Union[["WinLoss_0", "WinLoss_1", 'WinLoss_2']]
        X = Union.drop(["WinLoss_0", "WinLoss_1", 'WinLoss_2', 'Unnamed: 0'], axis=1)

        # 定義最佳化器
        optimizer = torch.optim.Adam(net.parameters(), lr)
        # optimizer = getattr(optim, OptiModal)(net.parameters(), lr)

        # for i in net.state_dict():
        #     print(i,":",net.state_dict()[i])
        # exit()

        losses = []  # 定义列表，用于接收每一步的损失值(train)
        lr_list = []  # 定義列表，用於接收每一步的學習率
        lossvalid = []  # 定义列表，用于接收每一epoch的损失值(valid)
        validacc = []

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,
                                                               verbose=False,
                                                               threshold=0.0001, threshold_mode='abs', cooldown=0,
                                                               min_lr=0,
                                                               eps=1e-08)

        for epoch in range(epochs):
            # 時間序列資料的交叉驗證 (依時序進行驗證)
            # -----------------------------------
            # 以變數 period 為基準進行劃分（從 0 到 3 為訓練資料，4 為測試資料）
            # 將變數 period 為 1, 2, 3 的資料作為驗證資料，比驗證資料更早以前的資料則作為訓練資料
            tss = TimeSeriesSplit(n_splits=4)

            for tr_idx, va_idx in tss.split(X):
                tr_x, va_x = X.iloc[tr_idx].to_numpy(), X.iloc[va_idx].to_numpy()
                tr_y, va_y = Y.iloc[tr_idx].to_numpy(), Y.iloc[va_idx].to_numpy()

                # 歸一化
                x_train_m, y_train_m = self.normalized(tr_x, tr_y)
                x_valid_m, y_valid_m = self.normalized(va_x, va_y)

                # print(x_valid_m,y_valid_m)
                # exit()
                # 返回的tensor和原tensor在梯度上或者数据上没有任何关系(將輸入的樣本標籤轉為張量)
                x_train_tensor = torch.from_numpy(x_train_m).type(torch.FloatTensor)
                y_train_tensor = torch.from_numpy(y_train_m).type(torch.FloatTensor)
                # valid
                x_valid_tensor = torch.from_numpy(x_valid_m).type(torch.FloatTensor)
                y_valid_tensor = torch.from_numpy(y_valid_m).type(torch.FloatTensor)
                # y_train_m,y_valid_m計算準確度

                net.train()  # 切换至训练模式
                # out_probs_train = net(X_train)
                loss = net.getloss(x_train_tensor, y_train_tensor)
                losses.append(loss.item())  # 保存中間狀態的損失值
                scheduler.step(loss.item())  # 呼叫學習率衰減物件
                optimizer.zero_grad()  # 清空之前的梯度
                loss.backward()  # 反向传播损失值
                optimizer.step()  # 更新参数
                # print(optimizer.state_dict())
                # exit()
                lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
                if epoch % 20 == 0:
                    print('Epoch {}/{} => Loss: {:.2f}'.format(epoch + 1, epochs, loss.item()))

                # x_train_tensor, y_train_tensor, x_valid_tensor,y_valid_tensor,y_train_m,y_valid_m
                # X_train, Y_train, X_valid, Y_valid, y_train, y_valid

                # 驗證模型
                # 如果不在意显存大小和计算时间的话，仅仅使用model.eval()已足够得到正确的validation的结果；而with torch.zero_grad()则是更进一步加速和节省gpu空间（因为不用计算和存储gradient）
                net.eval()  # 不启用 BatchNormalization 和 Dropout。此时 pytorch 会自动把 BN 和 DropOut 固定住，不会取平均，而是用训练好的值。不然的话，一旦 test 的 batch_size 过小，很容易就会因 BN 层导致模型 性能损失较大；
                with torch.no_grad():
                    output = net(x_valid_tensor)
                    loss_valid = net.getloss(x_valid_tensor, y_valid_tensor)
                    LossValue = loss_valid.item()  # 單個fold的loss
                    out_probs_valid = output.detach().numpy()
                    valid_acc = out_probs_valid.argmax(axis=1) == y_valid_m.argmax(axis=1)
                    print("Valid Accuracy:", float(np.mean(valid_acc)))  # 計算單個fold的準確率
                    foldValidACC = float(np.mean(valid_acc))
                    lossvalid.append(LossValue)
                    validacc.append(foldValidACC)
        # 儲存訓練資訊(PT)
        state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epochs}
        # print(state)

        torch.save(state,
                   "{}/BPNN_adam({}.{}_{}_{}_{}.{}).pt".format(folder, dropout1,
                                                               dropout2, lr,
                                                               epochs,
                                                               hidden_1,
                                                               hidden_2))

        # info所return的個數為n_trail的個數
        info = {
            'Forlder': folder,
            'dropout1': dropout1,
            'dropout2': dropout2,
            'learning_rate': lr,
            'epochs': epochs,
            'unit1': hidden_1,
            'unit2': hidden_2,
            'avg_loss': np.mean(lossvalid),  # 平均每個epoch
            'Valid Accuracy': np.mean(validacc)}  # cross-entropy平均每個epoch

        ValidAcc = float(np.mean(validacc))
        LossValid = float(np.mean(lossvalid))

        return ValidAcc, LossValid, info


def objective(trial):
    torch.manual_seed(0)
    dropout1 = trial.suggest_float("dropout1", 0, 1)
    dropout2 = trial.suggest_float("dropout2", 0, 1)
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-1)  # 學習率
    epochs = trial.suggest_int("epochs", 50, 300)  # 定義迭代次數
    unit1 = trial.suggest_int("unit1", 1, 160)  # 隱藏層個數
    unit2 = trial.suggest_int("unit2", 1, 160)  # 隱藏層個數
    # OptiModal = trial.suggest_categorical("OptiModal", ["Adam", "RMSprop", "SGD"])

    net = ThreelinearModel(unit1, unit2, dropout1, dropout2)  # 初始化网络
    modal = PytorchAdam()
    modal.path_num()
    ValidAccuracy, loss, info = modal.train('UnionTrain.csv', net, lr, epochs, unit1, unit2, dropout1, dropout2)
    AccData.append(info)
    print("----------Find Best modal----------")
    # modal.save(data)
    return loss, ValidAccuracy  # 需要設return，以利optuna尋找合適值最大值，需要在optuna.create_study設定


def SecondValid(times, Filename, lr, epoch, unit1, unit2, dropout1, dropout2, folder):
    print('======================')
    print(folder)
    print('======================')
    path_adam = "{}/BPNN_adam({}.{}_{}_{}_{}.{}).pt".format(folder, dropout1,
                                                            dropout2, lr,
                                                            epoch,
                                                            unit1,
                                                            unit2)
    print(path_adam)
    print('======================')
    if os.path.exists(path_adam):
        net = ThreelinearModel(unit1, unit2, dropout1, dropout2)
        modal = PytorchAdam()
        Union_Test = pd.read_csv(Filename, delimiter=",")
        Y = Union_Test[["WinLoss_0", "WinLoss_1", 'WinLoss_2']].to_numpy()
        X = Union_Test.drop(["WinLoss_0", "WinLoss_1", 'WinLoss_2', 'Unnamed: 0'], axis=1).to_numpy()

        # 歸一化
        x_test_m, y_test_m = modal.normalized(X, Y)
        # print(x_valid_m,y_valid_m)
        # exit()
        # 返回的tensor和原tensor在梯度上或者数据上没有任何关系(將輸入的樣本標籤轉為張量)
        x_test_tensor = torch.from_numpy(x_test_m).type(torch.FloatTensor)

        checkpoint = torch.load(path_adam)
        net.load_state_dict(checkpoint['model'])
        # optimizer = getattr(optim, OptiModal)(net.parameters(), lr)
        optimizer = torch.optim.Adam(net.parameters(), lr)
        optimizer.load_state_dict(checkpoint['optimizer'])
        net.eval()  # 評估模式
        output = net(x_test_tensor)  # predic_y 張量形式
        out_probs_test = output.detach().numpy()  # predic_y array形式
        SecondValid_acc = out_probs_test.argmax(axis=1) == Y.argmax(axis=1)
        print(out_probs_test.argmax(axis=1))
        print('=====================')
        print(Y.argmax(axis=1))
        print('=====================')
        print("SecondValid Accuracy{}:".format(times), np.mean(SecondValid_acc))
        SecondValidinfo = {'SecondValidAccuracy{}'.format(times): np.mean(SecondValid_acc)}
        return SecondValidinfo

    else:
        sys.exit("Need to Train the Modal")


def Test(Filename, lr, epoch, unit1, unit2, dropout1, dropout2, folder, Best_lossValid, Best_AccValid):
    print('======================')
    print(folder)
    print('======================')
    path_adam = "{}/BPNN_adam({}.{}_{}_{}_{}.{}).pt".format(folder, dropout1,
                                                            dropout2, lr,
                                                            epoch,
                                                            unit1,
                                                            unit2)
    print(path_adam)
    print('======================')
    if os.path.exists(path_adam):

        net = ThreelinearModel(unit1, unit2, dropout1, dropout2)
        modal = PytorchAdam()
        Union_Test = pd.read_csv(Filename, delimiter=",")
        Y = Union_Test[["WinLoss_0", "WinLoss_1", 'WinLoss_2']].to_numpy()
        X = Union_Test.drop(["WinLoss_0", "WinLoss_1", 'WinLoss_2', 'Unnamed: 0'], axis=1).to_numpy()

        # 歸一化
        x_test_m, y_test_m = modal.normalized(X, Y)
        # print(x_valid_m,y_valid_m)
        # exit()
        # 返回的tensor和原tensor在梯度上或者数据上没有任何关系(將輸入的樣本標籤轉為張量)
        x_test_tensor = torch.from_numpy(x_test_m).type(torch.FloatTensor)

        checkpoint = torch.load(path_adam)
        net.load_state_dict(checkpoint['model'])
        # optimizer = getattr(optim, OptiModal)(net.parameters(), lr)
        optimizer = torch.optim.Adam(net.parameters(), lr)
        optimizer.load_state_dict(checkpoint['optimizer'])
        net.eval()  # 評估模式
        output = net(x_test_tensor)  # predic_y 張量形式
        out_probs_test = output.detach().numpy()  # predic_y array形式
        test_acc = out_probs_test.argmax(axis=1) == Y.argmax(axis=1)
        print(out_probs_test.argmax(axis=1))
        print('=====================')
        print(Y.argmax(axis=1))
        print('=====================')
        print("Test Accuracy:", np.mean(test_acc))
        testinfo = {'TestAccuracy': np.mean(test_acc)}
        infotest = {
            'path_adam': path_adam,
            'dropout1': dropout1,
            'dropout2': dropout2,
            'learning_rate': lr,
            'epochs': epoch,
            'unit1': unit1,
            'unit2': unit2,
            'Best_lossValid': Best_lossValid,
            'Best_AccValid': Best_AccValid}
        infotest.update(testinfo)
        return infotest

    else:
        sys.exit("Need to Train the Modal")


def main(n_trials, mian_current_number):
    st = time.time()
    study = optuna.create_study(
        directions=["minimize", "maximize"],
        sampler=optuna.samplers.TPESampler(),
        storage='sqlite:///CPBL{}.db.sqlite3'.format(mian_current_number),  # 每個都是獨立一個網頁資料
        # storage='sqlite:///CPBL.db.sqlite3',  # 把每次跑的資料放在同一個儲存空間
        study_name='CPBL{}'.format(mian_current_number)
    )

    study.optimize(objective, n_trials)  # 要运行超参数调整，我们需要实例化一个study会话，调用optimize方法，并将我们的objective函数作为参数传递
    print("Number of finished trials:", len(study.trials))
    print('==========================')
    print('Best params:', study.best_trials)

    print('Best params:', study.best_trials[0].params)
    print('Best value: ', study.best_trials[0].values)  # 超参数调整过程完成后，我们可以通过访问best_trial方法来获取超参数的最佳组合
    OptimumLossValid = {'Best_lossValid': study.best_trials[0].values[0]}
    OptimumAccValid = {'Best_AccValid': study.best_trials[0].values[1]}
    Best_lossValid = study.best_trials[0].values[0]
    Best_AccValid = study.best_trials[0].values[1]
    # print('==========================')
    # print('best_params:', study.best_trials[0].params.items())
    # print('==========================')
    # print('best_params:', study.best_trials[0].params)
    # print('==========================')
    # for key, value in study.best_trials[0].params.items():
    #     print("{}: {}".format(key, value))
    # # print('==========================')
    # # print('best_trial:', study.best_trials[0])
    print('==========================')
    print('Time', time.time() - st)
    # bestvalue = {'BestValue': study.best_value}
    bestinfo = study.best_trials[0].params
    bestinfo.update(OptimumLossValid)
    bestinfo.update(OptimumAccValid)
    # print(bestinfo)
    # exit()

    OptimumDropout1 = study.best_trials[0].params['dropout1']
    OptimumDropout2 = study.best_trials[0].params['dropout2']
    OptimumEpochs = study.best_trials[0].params['epochs']
    OptimumLearningRate = study.best_trials[0].params['learning_rate']
    OptimumUnit1 = study.best_trials[0].params['unit1']
    OptimumUnit2 = study.best_trials[0].params['unit2']
    # OptiModal = study.best_trial.params['OptiModal']

    print(f"Number of trials on the Pareto front: {len(study.best_trials)}")

    # trial_with_highest_accuracy = max(study.best_trials, key=lambda t: t.values[i])
    # print(f"Trial with highest accuracy:")
    # print(f"\tnumber:{trial_with_highest_accuracy.number}")
    # print(f"\tparams:{trial_with_highest_accuracy.params}")
    # print(f"\tvalues:{trial_with_highest_accuracy.values}")

    pareto_front = optuna.visualization.plot_pareto_front(study, target_names=["LossValid",
                                                                               "ValidAccuracy"])  # 可视化地检查位于帕累托前沿上的 trials
    plotly.offline.plot(pareto_front, filename='pareto_front{}.html'.format(mian_current_number), auto_open=False)

    # importance = optuna.visualization.plot_param_importances(study,target=lambda t: t.values[0])  # 超参数的重要性
    # plotly.offline.plot(importance, filename='importance{}'.format(mian_current_number), auto_open=False)
    # history = optuna.visualization.plot_optimization_history(study,target=lambda t: t.values[0])  # 優化歷史，预期的行为是模型性能随着搜索次数的增加而提高
    # plotly.offline.plot(history, filename='history{}'.format(mian_current_number), auto_open=False)
    # slices = optuna.visualization.plot_slice(study,target=lambda t: t.values[0])  # 查看不同的单个超参数在多次试验中的变化情况，颜色对应试验次数。
    # plotly.offline.plot(slices, filename='slices{}'.format(mian_current_number), auto_open=False)
    return OptimumDropout1, OptimumDropout2, OptimumEpochs, OptimumLearningRate, OptimumUnit1, OptimumUnit2, bestinfo, Best_lossValid, Best_AccValid


def save_train(data, mian_current_number):
    filepath = 'C:/Users/mandy chou/Documents/GitHub/BPN_CPBL/TrainData/TrainACC{}.csv'.format(mian_current_number)
    # filename = 'TrainACC{}.csv'.format(mian_current_number)
    header = ['Forlder', 'dropout1', 'dropout2', 'epochs', 'learning_rate', 'unit1', 'unit2', 'avg_loss',
              'Valid Accuracy']
    with open(filepath, 'w', newline='') as f:
        writedCsv = csv.DictWriter(f, header)
        writedCsv.writeheader()
        writedCsv.writerows(data)
    f.close()


def save_test(data):
    path = glob.glob(os.path.join("TestData", "*"))
    SaveTestNumber = []
    for i in path:
        directory, file_name = os.path.split(i)
        pattern = '(\d+)'
        num = int(re.findall(pattern, file_name)[-1])
        SaveTestNumber.append(num)

    if len(SaveTestNumber) == 0:
        maxnum = 1
    else:
        maxnum = max(SaveTestNumber) + 1  # 統計目前使用文件最大編號(會在創建新資料夾之前的筆數)

    filepath = 'C:/Users/mandy chou/Documents/GitHub/BPN_CPBL/TestData/TestACC{}.csv'.format(maxnum)
    # filename = 'TestACC{}.csv'.format(maxnum)
    header = ['ForlderNum', 'NTrial', 'path_adam', 'dropout1', 'dropout2', 'epochs', 'learning_rate', 'unit1', 'unit2',
              'Best_lossValid', 'Best_AccValid', 'SecondValidAccuracy1', 'SecondValidAccuracy2', 'TestAccuracy']
    with open(filepath, 'w', newline='') as f:
        writedCsv = csv.DictWriter(f, header)
        writedCsv.writeheader()
        writedCsv.writerows(data)
    f.close()


if __name__ == "__main__":
    path_file = "C:/Users/mandy chou/Documents/GitHub/BPN_CPBL/TestACC13.csv"  # ★★★★★★★★★每個folder的最優解
    root, file_name = os.path.split(path_file)  # C:/Users/mandy chou/Documents/GitHub/BPN_CPBL
    if os.path.exists(path_file):
        TransformData("6_v1.csv")  # ★★★★★★★★★需要確認是哪一個檔案下的內容
        fp = open("TestACC13.csv", "r", encoding="utf-8")  # ★★★★★★★★★每個folder的最優解
        csv_reader = csv.reader(fp)
        data = list(csv_reader)
        fp.close()
        # data[0]即為表格的標題列，data[1]之後的內容即為實際資料的內容
        # Forlder	dropout1	dropout2	epochs	learning_rate	unit1	unit2
        SaveTestData = []
        for d in data[1:]:
            path=d[2]
            pattern = re.compile('([0-9]+\.[0-9]+)',re.S)
            r_list = pattern.findall(path)

            pattern2 = re.compile('/folder([0-9]+)',re.S)
            folder_num = pattern2.findall(path)
            # print(path)
            # print(folder_num)
            # # print(r_list[0])
            # exit()
            path_adam = 'C:/Users/mandy chou/Documents/GitHub/BPN_CPBL/PTFile/folder{}'.format(int(folder_num[0]))
            OptimumDropout1 = float(r_list[0])
            OptimumDropout2 = float(r_list[1])
            OptimumEpochs = int(d[5])
            OptimumLearningRate = float(r_list[2])
            OptimumUnit1 = int(d[7])
            OptimumUnit2 = int(d[8])
            Best_lossValid = float(d[9])
            Best_AccValid = float(d[10])
            singleinfo = {}
            info = SecondValid(1, 'UnionValid.csv', OptimumLearningRate, OptimumEpochs, OptimumUnit1,
                               OptimumUnit2,
                               OptimumDropout1,
                               OptimumDropout2, path_adam)
            info2 = SecondValid(2, 'UnionValid2.csv', OptimumLearningRate, OptimumEpochs, OptimumUnit1,
                                OptimumUnit2,
                                OptimumDropout1,
                                OptimumDropout2, path_adam)
            testacc = Test('UnionTest.csv', OptimumLearningRate, OptimumEpochs, OptimumUnit1, OptimumUnit2,
                           OptimumDropout1,
                           OptimumDropout2, path_adam, Best_lossValid, Best_AccValid)
            singleinfo.update(info)
            singleinfo.update(info2)
            singleinfo.update(testacc)
            SaveTestData.append(singleinfo)
        save_test(SaveTestData)


    else:
        for i in range(1, 2):
            TransformData("6_v{}.csv".format(i))
            BestData = []
            for k in range(850,1501,50):
                AccData = []
                n_trials = k
                path = glob.glob(os.path.join("PTFile", "*"))  # PTFile\folder1 #PTFile\folder2

                UseNumber = []
                mian_current_number = 0
                for i in path:
                    directory, file_name = os.path.split(i)
                    pattern = '(\d+)'
                    num = int(re.findall(pattern, file_name)[-1])
                    UseNumber.append(num)
                if len(UseNumber) == 0:
                    mian_current_number = 0
                else:
                    mian_current_number = max(UseNumber)  # 統計目前使用文件最大編號(會在創建新資料夾之前的筆數)
                mian_current_number += 1

                folder = "C:/Users/mandy chou/Documents/GitHub/BPN_CPBL/PTFile/folder{}".format(mian_current_number)
                os.mkdir(folder)  # 創建儲存.pt檔

                OptimumDropout1, OptimumDropout2, OptimumEpochs, OptimumLearningRate, OptimumUnit1, OptimumUnit2, bestinfo, Best_lossValid, Best_AccValid = main(
                    n_trials, mian_current_number)
                info = SecondValid(1, 'UnionValid.csv', OptimumLearningRate, OptimumEpochs, OptimumUnit1,
                               OptimumUnit2,
                               OptimumDropout1,
                               OptimumDropout2, folder)
                info2 = SecondValid(2, 'UnionValid2.csv', OptimumLearningRate, OptimumEpochs, OptimumUnit1,
                                OptimumUnit2,
                                OptimumDropout1,
                                OptimumDropout2, folder)
                TestAcc = Test('UnionTest.csv', OptimumLearningRate, OptimumEpochs, OptimumUnit1, OptimumUnit2,
                               OptimumDropout1,
                               OptimumDropout2, folder, Best_lossValid, Best_AccValid)
                n_trial = {'NTrial': k}
                ForlderNum = {'ForlderNum': mian_current_number}
                bestinfo.update(info)
                bestinfo.update(info2)
                bestinfo.update(TestAcc)
                bestinfo.update(n_trial)
                bestinfo.update(ForlderNum)
                BestData.append(bestinfo)
                save_train(AccData, mian_current_number)

            save_test(BestData)
