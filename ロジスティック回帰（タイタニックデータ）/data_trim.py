import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
pd.set_option('display.max_rows', 1000000)
pd.set_option('display.max_columns', 1000000)


train_data = pd.read_csv("/Users/tsukamotoakihiko/Documents/programing/python/達人データサイエンティスト/自由作成/ロジスティック回帰（タイタニックデータ）/titanic_train.csv")
test_data = pd.read_csv("/Users/tsukamotoakihiko/Documents/programing/python/達人データサイエンティスト/自由作成/ロジスティック回帰（タイタニックデータ）/titanic_test.csv")

##train_dataには答えラベルの"Survived"列があるがtest_dataにはそれがない（当たり前やけど）
train_data.tail()
test_data.tail()

##ID列が有効な特徴量か独立性の検定を行う　→ 意味ある特徴量ではない（実行の必要なし）
# obs1 = sum(train_data.loc[0:300, "Survived"])
# obs2 = sum(train_data.loc[301:600, "Survived"])
# obs3 = sum(train_data.loc[601:, "Survived"])
# ##結果　→ Chi2ContingencyResult(statistic=0.0, pvalue=1.0, dof=0, expected_freq=array([109., 191.]))
# result = stats.chi2_contingency([obs1, 300 - obs1], [obs2, 300 - obs2], [obs3, train_data.shape[0] - 600 - obs3])
# print(result)

##非優位なID列と活用が難しいName列は除去する
train_data.drop(["PassengerId", "Name", "Ticket"], axis = 1, inplace = True)  #inplaceは破壊的メソッドにするか否かの引数
test_data.drop(["PassengerId", "Name", "Ticket"], axis = 1, inplace = True)

##ラベル、特徴量に分ける
x_train = train_data.iloc[:, 1:train_data.shape[1]]
y_train = train_data.iloc[:, 0]
x_test = test_data


##pclassが1,2,3の順番に高級なクラスのようなのでその逆の3,2,1にする
for i in range(x_train.shape[0]):
    x_train.loc[i, "Pclass"] = abs(x_train.loc[i, "Pclass"] - 3) + 1

for i in range(x_test.shape[0]):
    x_test.loc[i, "Pclass"] = abs(x_test.loc[i, "Pclass"] - 3) + 1

##sex列についてカテゴリカルデータを数値データに変換する
for i in range(x_train.shape[0]):
    if(x_train.loc[i, "Sex"] == "male"):
        x_train.loc[i, "Sex"] = 1
    else:
        x_train.loc[i, "Sex"] = 0

for i in range(x_test.shape[0]):
    if(x_test.loc[i, "Sex"] == "male"):
        x_test.loc[i, "Sex"] = 1
    else:
        x_test.loc[i, "Sex"] = 0

##NANの処理
x_train.isnull().sum()
x_test.isnull().sum()

##AgeのNAN処理（Rのaggregate()のようなもので当てはめる）
x_train["Age"] = x_train["Age"].fillna(x_train.groupby(["Pclass","Sex"])["Age"].transform("mean"))
x_test["Age"] = x_test["Age"].fillna(x_train.groupby(["Pclass","Sex"])["Age"].transform("mean"))

##CabinのNAN処理（部屋がわかっているものを1、そうでないものを0として扱う）
for i in range(x_train.shape[0]):
    if pd.isna(x_train.loc[i, "Cabin"]):
        x_train.loc[i, "Cabin"] = 0
    else:
        x_train.loc[i, "Cabin"] = 1

for i in range(x_test.shape[0]):
    if pd.isna(x_test.loc[i, "Cabin"]):
        x_test.loc[i, "Cabin"] = 0
    else:
        x_test.loc[i, "Cabin"] = 1

##Embarked列のNANの処理（作ってはみたもののテストデータのEmbarked列に欠損値がなかった
# for i in range(train_data.shape[0]):
#     if pd.isna(x_train.loc[i, "Embarked"]):
#         x_train.loc[i, "Embarked"] = "no_record"

# print(np.unique(x_train.loc[:,"Embarked"]))  #Rのtable()に対応する関数

drop_ind = []
##欠損値も２個しかないので行ごと削除して対応した
for i in range(x_train.shape[0]):
    if pd.isna(x_train.loc[i, "Embarked"]):
        drop_ind.append(i)
x_train = x_train.drop(drop_ind)
y_train = y_train.drop(drop_ind)

##テストデータのfare列に欠損値があったので平均で埋め合わせる
Fare_mean = x_test.loc[:, "Fare"].mean()
for i in range(x_test.shape[0]):
    if pd.isna(x_test.loc[i, "Fare"]):
        x_test.loc[i, "Fare"] = Fare_mean

##Embarkedをone-hotコーディングする
x_train = pd.get_dummies(x_train, drop_first= True)
x_test = pd.get_dummies(x_test, drop_first= True)


##相関行列を求めてみる
cor_matrix = pd.concat([x_train, y_train], axis = 1).corr(method='pearson')
print(cor_matrix)
sns.heatmap(cor_matrix,
            vmin = -1.0,
            vmax = 1.0,
            center = 0,
            annot = True, # True:格子の中に値を表示
            fmt = '.5f',
            xticklabels = cor_matrix.columns.values,
            yticklabels = cor_matrix.columns.values
           )
plt.show()

##数値特徴量をmin-max正規化する
mms = MinMaxScaler()
x_train.loc[:, ["Pclass", "Age", "SibSp", "Parch", "Fare"]] = mms.fit_transform(x_train.loc[:, ["Pclass", "Age", "SibSp", "Parch", "Fare"]])
x_test.loc[:, ["Pclass", "Age", "SibSp", "Parch", "Fare"]] = mms.fit_transform(x_test.loc[:, ["Pclass", "Age", "SibSp", "Parch", "Fare"]])

#pickelファイルに変換する
with open('x_train', 'wb') as f:
    pickle.dump(x_train, f)

with open('y_train', 'wb') as f:
    pickle.dump(y_train, f)

with open('x_test', 'wb') as f:
    pickle.dump(x_test, f)