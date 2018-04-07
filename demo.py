from models import LSTM_FC_Model


X_test = None
model = LSTM_FC_Model

print('Loading model......')
model.load_model_params('checkpoints/LSTM_FC_10000')

print('Predicting......')
Y_output = model.predict(X_test)

print('Done!')