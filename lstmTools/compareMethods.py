from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, GRU, Dense, Dropout, Input

# 构建 RNN 模型
def build_RNN_model(input_shape, output_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(SimpleRNN(64, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(SimpleRNN(32, return_sequences=False))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_shape))
    model.compile(optimizer='adam', loss='mse')
    return model


# 构建 GRU 模型
def build_GRU_model(input_shape, output_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(GRU(64, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(GRU(32, return_sequences=False))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_shape))
    model.compile(optimizer='adam', loss='mse')
    return model