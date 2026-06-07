# -*- coding: utf-8 -*-

import csv
import glob
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import plotly
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna

from sklearn.model_selection import TimeSeriesSplit

BASE_DIR = r"C:\Users\mandy chou\PycharmProjects\CPBL_CrawlerBPN"
PT_DIR = os.path.join(BASE_DIR, "PTFile")
# TRAIN_DIR = os.path.join(BASE_DIR, "TrainData")
TEST_DIR = os.path.join(BASE_DIR, "TestData")

os.makedirs(PT_DIR, exist_ok=True)
# os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

torch.manual_seed(0)

TOP20 = [
    'PreBAT.WOBA',
    'PreBAT.BB%',
    'PreBAT.K%',
    'Brother.PreAP_SO',
    'PreAP_SO',
    'Brother.PreAP_BB',
    'Brother.PreFIP',
    'Brother.PreBAT.A_R',
    'PreAP_HR',
    'Brother.PreBABIP',
    'PreBAT.BABIP',
    'Brother.PitcherHabit_1',
    'PreBAT.BIP%',
    'Brother.PreBB%',
    'Brother.PreBAT.BABIP',
    'PreAP_BB',
    'Brother.PreWHIP',
    'PreERA',
    'Brother.PreBAT.BIP%',
    'Brother.PreAP_RunCnt'
]


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class BPNBinaryModel(nn.Module):
    def __init__(self, hidden1, hidden2, dropout1, dropout2):
        super().__init__()

        self.linear1 = nn.Linear(68, hidden1)
        self.dropout1 = nn.Dropout(dropout1)
        self.mish1 = Mish()

        self.linear2 = nn.Linear(hidden1, hidden2)
        self.dropout2 = nn.Dropout(dropout2)
        self.mish2 = Mish()

        self.linear3 = nn.Linear(hidden2, 2)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        out1 = self.mish1(self.dropout1(self.linear1(x)))
        out2 = self.mish2(self.linear2(self.dropout2(out1)))
        return self.linear3(out2)

    def getloss(self, x, y):
        y_pred = self.forward(x)
        return self.criterion(y_pred, y)


def TransformData(OriginData):
    Union = pd.read_csv(OriginData, delimiter=",")

    Union["IntPitcherHabit"] = Union["PitcherHabit"].astype(int)
    Union["IntBrother.PitcherHabit"] = Union["Brother.PitcherHabit"].astype(int)
    Union["IntVisitingHomeType"] = Union["VisitingHomeType"].astype(int)
    Union["IntWinLoss"] = Union["WinLoss"].astype(int)

    for col in Union.select_dtypes(include=["object"]).columns:
        if col not in ["GameDate", "DayNight"]:
            Union[col] = pd.factorize(Union[col])[0]

    Union = pd.concat([
        Union,
        pd.get_dummies(Union["DayNight"], prefix="DayNight"),
        pd.get_dummies(Union["IntPitcherHabit"], prefix="PitcherHabit"),
        pd.get_dummies(Union["IntBrother.PitcherHabit"], prefix="Brother.PitcherHabit"),
        pd.get_dummies(Union["IntVisitingHomeType"], prefix="VisitingHomeType"),
        pd.get_dummies(Union["IntWinLoss"], prefix="WinLoss")
    ], axis=1)

    UnionDF = Union.drop(
        [
            "GameDate", "MinusScore",
            "IntPitcherHabit", "PitcherHabit",
            "IntBrother.PitcherHabit", "Brother.PitcherHabit",
            "VisitingHomeType", "IntVisitingHomeType",
            "DayNight", "WinLoss", "IntWinLoss",
            "VisitingHomeType_2", "DayNight_N"
        ],
        axis=1,
        errors="ignore"
    )

    for col in ["WinLoss_0", "WinLoss_1", "WinLoss_2"]:
        if col not in UnionDF.columns:
            UnionDF[col] = 0

    UnionDF = UnionDF.astype(np.float32)

    UnionDF.iloc[:260, :].to_csv("UnionTrain.csv", index=False)
    UnionDF.iloc[260:, :].to_csv("UnionTest.csv", index=False)
    UnionDF.iloc[234:260, :].to_csv("UnionValid.csv", index=False)
    UnionDF.iloc[208:260, :].to_csv("UnionValid2.csv", index=False)
    UnionDF.to_csv("Union.csv", index=False)


def load_xy_binary(filename):
    df = pd.read_csv(filename)

    df = df[df["WinLoss_2"] != 1].copy()

    y = df["WinLoss_1"].astype(np.int64).to_numpy()

    X = df.drop(
        ["WinLoss_0", "WinLoss_1", "WinLoss_2", "Unnamed: 0"],
        axis=1,
        errors="ignore"
    )

    # X = X[TOP20].astype(np.float32).to_numpy()
    X = X.astype(np.float32).to_numpy()

    return X, y


class BPNTrainer:
    def __init__(self, folder):
        self.folder = folder
        os.makedirs(self.folder, exist_ok=True)

    def normalized(self, x_data):
        e = 1e-7
        x_data = x_data.astype(np.float32)

        for i in range(x_data.shape[1]):
            max_num = np.max(x_data[:, i])
            min_num = np.min(x_data[:, i])
            x_data[:, i] = (x_data[:, i] - min_num + e) / (max_num - min_num + e)

        return x_data

    def train(self, filename, net, lr, epochs, hidden1, hidden2, dropout1, dropout2):
        X, Y = load_xy_binary(filename)

        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            threshold=0.0001,
            threshold_mode="abs",
            cooldown=0,
            min_lr=0,
            eps=1e-08
        )

        lossvalid = []
        validacc = []

        for epoch in range(epochs):
            tss = TimeSeriesSplit(n_splits=4)

            for tr_idx, va_idx in tss.split(X):
                tr_x = X[tr_idx]
                va_x = X[va_idx]

                tr_y = Y[tr_idx]
                va_y = Y[va_idx]

                x_train_m = self.normalized(tr_x.copy())
                x_valid_m = self.normalized(va_x.copy())

                x_train_tensor = torch.from_numpy(x_train_m).float()
                y_train_tensor = torch.from_numpy(tr_y).long()

                x_valid_tensor = torch.from_numpy(x_valid_m).float()
                y_valid_tensor = torch.from_numpy(va_y).long()

                net.train()
                loss = net.getloss(x_train_tensor, y_train_tensor)

                scheduler.step(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if epoch % 20 == 0:
                    print(f"Epoch {epoch + 1}/{epochs} => Loss: {loss.item():.4f}")

                net.eval()
                with torch.no_grad():
                    output = net(x_valid_tensor)
                    loss_valid = net.getloss(x_valid_tensor, y_valid_tensor)

                    pred = output.detach().numpy().argmax(axis=1)
                    acc = np.mean(pred == va_y)

                    print("Valid Accuracy:", float(acc))

                    lossvalid.append(loss_valid.item())
                    validacc.append(float(acc))

        path_adam = os.path.join(
            self.folder,
            f"BPN_binary({dropout1}_{dropout2}_{lr}_{epochs}_{hidden1}_{hidden2}).pt"
        )

        state = {
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epochs
        }

        torch.save(state, path_adam)

        print("\nModel Saved:")
        print(path_adam)

        info = {
            "Forlder": self.folder,
            "path_adam": path_adam,
            "dropout1": dropout1,
            "dropout2": dropout2,
            "learning_rate": lr,
            "epochs": epochs,
            "unit1": hidden1,
            "unit2": hidden2,
            "avg_loss": np.mean(lossvalid),
            "Valid Accuracy": np.mean(validacc)
        }

        return float(np.mean(validacc)), float(np.mean(lossvalid)), info


def evaluate_saved_model(
        filename,
        path_adam,
        hidden1,
        hidden2,
        dropout1,
        dropout2,
        label_name):
    print("======================")
    print(path_adam)
    print(os.path.exists(path_adam))
    print("======================")

    if not os.path.exists(path_adam):
        sys.exit(f"Model Not Found:\n{path_adam}")

    X, Y = load_xy_binary(filename)

    trainer = BPNTrainer(
        os.path.dirname(path_adam)
    )

    X_m = trainer.normalized(
        X.copy()
    )

    x_tensor = torch.from_numpy(
        X_m
    ).float()

    net = BPNBinaryModel(
        hidden1,
        hidden2,
        dropout1,
        dropout2
    )

    checkpoint = torch.load(
        path_adam,
        map_location="cpu"
    )

    net.load_state_dict(
        checkpoint["model"]
    )

    net.eval()

    with torch.no_grad():
        output = net(x_tensor)

        pred = output.detach().numpy().argmax(axis=1)

    acc = np.mean(pred == Y)

    print("\n======================")
    print(label_name)
    print("======================")
    print("Pred:", pred)
    print("True:", Y)
    print("Accuracy:", acc)

    return {
        f"{label_name}Accuracy": float(acc)
    }


def objective(trial):
    global AccData
    global CURRENT_FOLDER

    torch.manual_seed(0)

    dropout1 = trial.suggest_float("dropout1", 0, 0.5)
    dropout2 = trial.suggest_float("dropout2", 0, 0.5)
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2)
    epochs = trial.suggest_int("epochs", 50, 150)
    unit1 = trial.suggest_int("unit1", 4, 64)
    unit2 = trial.suggest_int("unit2", 4, 64)

    net = BPNBinaryModel(unit1, unit2, dropout1, dropout2)

    trainer = BPNTrainer(CURRENT_FOLDER)

    valid_acc, loss, info = trainer.train(
        "UnionTrain.csv",
        net,
        lr,
        epochs,
        unit1,
        unit2,
        dropout1,
        dropout2
    )

    AccData.append(info)

    return loss, valid_acc


def run_optuna(n_trials, folder_number):
    global CURRENT_FOLDER

    CURRENT_FOLDER = os.path.join(PT_DIR, f"folder{folder_number}")
    os.makedirs(CURRENT_FOLDER, exist_ok=True)

    st = time.time()

    study = optuna.create_study(
        directions=["minimize", "maximize"],
        sampler=optuna.samplers.TPESampler(seed=0),
        storage=f"sqlite:///BPN_binary_{folder_number}.db.sqlite3",
        study_name=f"BPN_binary_{folder_number}",
        load_if_exists=True
    )

    study.optimize(objective, n_trials=n_trials)

    best_trial = study.best_trials[0]

    print("Number of finished trials:", len(study.trials))
    print("Best params:", best_trial.params)
    print("Best values:", best_trial.values)
    print("Time:", time.time() - st)

    pareto_front = optuna.visualization.plot_pareto_front(
        study,
        target_names=["LossValid", "ValidAccuracy"]
    )

    plotly.offline.plot(
        pareto_front,
        filename=f"pareto_front_binary_{folder_number}.html",
        auto_open=False
    )

    bestinfo = best_trial.params.copy()
    bestinfo.update({
        "Best_lossValid": best_trial.values[0],
        "Best_AccValid": best_trial.values[1]
    })

    return (
        best_trial.params["dropout1"],
        best_trial.params["dropout2"],
        best_trial.params["epochs"],
        best_trial.params["learning_rate"],
        best_trial.params["unit1"],
        best_trial.params["unit2"],
        bestinfo,
        best_trial.values[0],
        best_trial.values[1],
        CURRENT_FOLDER
    )


def save_result(data):
    path = glob.glob(os.path.join(TEST_DIR, "BPNBinaryACC*.csv"))

    nums = []

    for p in path:
        _, file_name = os.path.split(p)
        found = re.findall(r"(\d+)", file_name)
        if found:
            nums.append(int(found[-1]))

    if len(nums) == 0:
        maxnum = 1
    else:
        maxnum = max(nums) + 1

    filepath = os.path.join(TEST_DIR, f"BPNBinaryACC{maxnum}.csv")

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False, encoding="utf-8-sig")

    print("已輸出：", filepath)


def predict_by_best_model(csv_file, result_file):
    TransformData(csv_file)

    df = pd.read_csv(result_file)

    row = df.iloc[0]

    path_adam = row["path_adam"]

    result = evaluate_saved_model(
        "UnionTest.csv",
        path_adam,
        int(row["unit1"]),
        int(row["unit2"]),
        float(row["dropout1"]),
        float(row["dropout2"]),
        "Predict"
    )

    output_file = os.path.join(
        TEST_DIR,
        "PredictResult.csv"
    )

    pd.DataFrame([result]).to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n========== Prediction Result ==========")
    print(result)
    print("已輸出：", output_file)


if __name__ == "__main__":

    csv_file = os.path.join(
        BASE_DIR,
        "6_v1.csv"
    )

    result_file = os.path.join(
        TEST_DIR,
        "BPNBinaryACC1.csv"
    )

    if os.path.exists(result_file):

        print("找到最佳模型，直接預測")

        predict_by_best_model(
            csv_file,
            result_file
        )

    else:

        print("找不到最佳模型，開始訓練")

        TransformData(csv_file)

        AccData = []

        folder_number = 1

        (
            OptimumDropout1,
            OptimumDropout2,
            OptimumEpochs,
            OptimumLearningRate,
            OptimumUnit1,
            OptimumUnit2,
            bestinfo,
            Best_lossValid,
            Best_AccValid,
            folder
        ) = run_optuna(
            100,
            folder_number
        )

        # 建立模型路徑
        path_adam = os.path.join(
            folder,
            f"BPN_binary({OptimumDropout1}_{OptimumDropout2}_{OptimumLearningRate}_{OptimumEpochs}_{OptimumUnit1}_{OptimumUnit2}).pt"
        )

        valid_info = evaluate_saved_model(
            "UnionValid.csv",
            path_adam,
            OptimumUnit1,
            OptimumUnit2,
            OptimumDropout1,
            OptimumDropout2,
            "Valid"
        )

        valid2_info = evaluate_saved_model(
            "UnionValid2.csv",
            path_adam,
            OptimumUnit1,
            OptimumUnit2,
            OptimumDropout1,
            OptimumDropout2,
            "Valid2"
        )

        test_info = evaluate_saved_model(
            "UnionTest.csv",
            path_adam,
            OptimumUnit1,
            OptimumUnit2,
            OptimumDropout1,
            OptimumDropout2,
            "Test"
        )

        result = {
            "path_adam": path_adam,
            "dropout1": OptimumDropout1,
            "dropout2": OptimumDropout2,
            "learning_rate": OptimumLearningRate,
            "epochs": OptimumEpochs,
            "unit1": OptimumUnit1,
            "unit2": OptimumUnit2,
            "Best_lossValid": Best_lossValid,
            "Best_AccValid": Best_AccValid
        }

        result.update(valid_info)
        result.update(valid2_info)
        result.update(test_info)

        save_result([result])

        print("\n===== 最終結果 =====")
        print(result)