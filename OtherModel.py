# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


# =========================
# 1. 沿用你的資料轉換
# =========================

def TransformData(OriginData):
    Union = pd.read_csv(OriginData, delimiter=",")

    Union['IntPitcherHabit'] = Union['PitcherHabit'].astype(int)
    Union['IntBrother.PitcherHabit'] = Union['Brother.PitcherHabit'].astype(int)
    Union['IntVisitingHomeType'] = Union['VisitingHomeType'].astype(int)
    Union['IntWinLoss'] = Union['WinLoss'].astype(int)

    Union = pd.concat([
        Union,
        pd.get_dummies(Union['DayNight'], prefix='DayNight'),
        pd.get_dummies(Union['IntPitcherHabit'], prefix='PitcherHabit'),
        pd.get_dummies(Union['IntBrother.PitcherHabit'], prefix='Brother.PitcherHabit'),
        pd.get_dummies(Union['IntVisitingHomeType'], prefix='VisitingHomeType'),
        pd.get_dummies(Union['IntWinLoss'], prefix='WinLoss')
    ], axis=1)

    UnionDF = Union.drop(
        [
            'GameDate', 'MinusScore',
            'IntPitcherHabit', 'PitcherHabit',
            'IntBrother.PitcherHabit', 'Brother.PitcherHabit',
            'VisitingHomeType', 'IntVisitingHomeType',
            'DayNight', 'WinLoss', 'IntWinLoss',
            'VisitingHomeType_2', 'DayNight_N'
        ],
        axis=1,
        errors="ignore"
    )

    # 確保三個 label 欄位都存在
    for col in ["WinLoss_0", "WinLoss_1", "WinLoss_2"]:
        if col not in UnionDF.columns:
            UnionDF[col] = 0

    UnionDF_Data = UnionDF.iloc[:260, :]
    UnionDF_Exam = UnionDF.iloc[260:, :]
    UnionDF_Valid = UnionDF.iloc[234:260, :]
    UnionDF_Valid2 = UnionDF.iloc[208:260, :]

    UnionDF.to_csv("Union.csv", index=False)
    UnionDF_Data.to_csv("UnionTrain.csv", index=False)
    UnionDF_Exam.to_csv("UnionTest.csv", index=False)
    UnionDF_Valid.to_csv("UnionValid.csv", index=False)
    UnionDF_Valid2.to_csv("UnionValid2.csv", index=False)


# =========================
# 2. 載入資料：改成二分類
# =========================

def load_xy_binary(filename):
    df = pd.read_csv(filename)

    # 移除 WinLoss_2，也就是第三類
    df = df[df["WinLoss_2"] != 1].copy()

    # y: WinLoss_1 當作 1，WinLoss_0 當作 0
    y = df["WinLoss_1"].astype(int).values

    X = df.drop(
        ["WinLoss_0", "WinLoss_1", "WinLoss_2", "Unnamed: 0"],
        axis=1,
        errors="ignore"
    )

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

    # X = X[TOP20]

    return X, y


def check_class_count(filename):
    X, y = load_xy_binary(filename)

    print("\n==============================")
    print(filename)
    print("資料筆數:", len(y))
    print(pd.Series(y).value_counts().sort_index())
    print(pd.Series(y).value_counts(normalize=True).sort_index())


# =========================
# 3. 評估模型
# =========================

def evaluate_model(model, train_file, test_file, model_name):
    X_train, y_train = load_xy_binary(train_file)
    X_test, y_test = load_xy_binary(test_file)

    model.fit(X_train, y_train)
    # 只印 XGBoost 的特徵重要度
    if "XGBoost" in model_name and hasattr(model, "feature_importances_"):
        importance = pd.DataFrame({
            "feature": X_train.columns,
            "importance": model.feature_importances_
        })

        importance = importance.sort_values(
            "importance",
            ascending=False
        )

        print("\n===== Top 20 Features =====")
        print(importance.head(20))

    pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        pred_prob = model.predict_proba(X_test)
        loss = log_loss(y_test, pred_prob, labels=[0, 1])
    else:
        loss = None

    acc = accuracy_score(y_test, pred)

    print("\n==============================")
    print(model_name)
    print("Train:", train_file)
    print("Test :", test_file)
    print("Accuracy:", acc)

    if loss is not None:
        print("LogLoss:", loss)

    print("Pred:", pred)
    print("True:", y_test)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, pred, labels=[0, 1]))

    print("\nClassification Report:")
    print(classification_report(y_test, pred, labels=[0, 1], zero_division=0))

    return {
        "model": model_name,
        "test_file": test_file,
        "accuracy": acc,
        "logloss": loss
    }


# =========================
# 4. 建立模型
# =========================

def get_models():
    models = {}

    models["Logistic Regression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=0
        ))
    ])

    models["Random Forest"] = RandomForestClassifier(
        n_estimators=100,
        max_depth=3,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=0
    )

    models["XGBoost"] = XGBClassifier(
        n_estimators=60,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=2,
        reg_alpha=1,
        reg_lambda=10,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=0,
        tree_method="hist"
    )

    return models


# =========================
# 5. 主程式
# =========================

if __name__ == "__main__":

    # 第一次執行或原始資料有更新時要跑
    # TransformData("6_v1.csv")

    # X, y = load_xy_binary("UnionTrain.csv")
    #
    # print("\n===== Feature List =====")
    # for i, col in enumerate(X.columns, start=1):
    #     print(f"{i}. {col}")

    print("\n========== 類別分布 ==========")
    check_class_count("UnionTrain.csv")
    check_class_count("UnionValid.csv")
    check_class_count("UnionValid2.csv")
    check_class_count("UnionTest.csv")

    models = get_models()

    results = []

    for model_name, model in models.items():
        results.append(
            evaluate_model(
                model=model,
                train_file="UnionTrain.csv",
                test_file="UnionValid.csv",
                model_name=model_name + " - Valid"
            )
        )

        results.append(
            evaluate_model(
                model=model,
                train_file="UnionTrain.csv",
                test_file="UnionValid2.csv",
                model_name=model_name + " - Valid2"
            )
        )

        results.append(
            evaluate_model(
                model=model,
                train_file="UnionTrain.csv",
                test_file="UnionTest.csv",
                model_name=model_name + " - Test"
            )
        )

    result_df = pd.DataFrame(results)

    print("\n========== 總結 ==========")
    print(result_df)

    result_df.to_csv("model_compare_result.csv", index=False, encoding="utf-8-sig")
    print("\n已輸出：model_compare_result.csv")
