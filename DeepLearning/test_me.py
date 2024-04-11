import numpy as np
import pandas as pd
from tsai.all import *
from sklearn.model_selection import train_test_split

# 1단계: 임의의 OHLCV 데이터 생성
np.random.seed(0)
dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
data = np.random.rand(1000, 5) * np.array([100, 105, 95, 100, 1000])[None, :]
ohlcv = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'], index=dates)

# 데이터 분할 (학습용 80%, 테스트용 20%)
X_train, X_test, y_train, y_test = train_test_split(ohlcv.index.values, ohlcv['open'].values, test_size=0.2, random_state=0)

# 2단계: TSDatasets 준비
splits = (list(range(len(X_train))), list(range(len(X_train), len(X_train) + len(X_test))))
X = ohlcv[['open', 'high', 'low', 'close', 'volume']].values
y = ohlcv['open'].values
tfms = [None, TSForecasting()]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)

# 3단계: Learner 객체 준비 및 모델 학습
# patchTST 모델 사용
batch_tfms = TSStandardize(by_sample=True)
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], batch_tfms=batch_tfms, shuffle_train=True)
model = PatchTST(dls.vars, dls.c, seq_len=dls.len)
learn = Learner(dls, model, metrics=mae)
learn.fit_one_cycle(10, lr_max=1e-3)
