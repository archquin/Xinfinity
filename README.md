# Xinfinity
LSTM for predicting exchange rates

(written in python 3.10)
Dependencies
- Tensorflow
- Keras LSTM,DENSE, layer, Sequential
- Sklearn,processing min_max_error
- Sklearn.metrics MSE metric
- Sklearn.model_selection train_test_split (function)
- yfinance

Tensors were manually set.

- Input: train set for X days from yfinance, test set i.e. remaints X after test set series, data X2 fit from yfinance that is fit with X data with LSTM NN
- Ouptu: for test set that remains after X, an elaborate predction for the upcoming data as in X prediction according to X--> X2 fit.

1) Creates training and test set, and also inquires X2 fit set for X data
2) Creates tensors for horizontal regression for X and X2
3) Utilizes LSTM model that makes fit of X data to X2
4) Makes prediction of X test, i.e. rate that s going to be tomorrow according to data!
(after having fed X trained before for X2 test now, X now gives X2 predicament)
