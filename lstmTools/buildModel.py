from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Layer, MultiHeadAttention, RepeatVector, Input
from tensorflow.keras.layers import LayerNormalization, Add
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import SimpleRNN, GRU, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def compute_output_shape(self, input_shape):
        return input_shape
class MultiHeadAttentionLayer(Layer):
    def __init__(self, num_heads, key_dim, **kwargs):
        super(MultiHeadAttentionLayer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

    def call(self, inputs):
        attn_output = self.mha(inputs, inputs)
        return attn_output

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], input_shape[-1]), initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[-1],), initializer='zeros', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        score = K.tanh(K.dot(inputs, self.W) + self.b)
        attention_weights = K.softmax(score, axis=1)
        context_vector = attention_weights * inputs
        context_vector = K.sum(context_vector, axis=1)
        return context_vector

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

# 构建 LSTM 模型
def build_LSTM_model(input_shape, output_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_shape))
    model.compile(optimizer='adam', loss='mse')
    return model

# 构建 多头注意力 LSTM 模型
def build_LSTM_Multi_Attention_model(input_shape, output_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(32, return_sequences=True))
    model.add(MultiHeadAttentionLayer(num_heads=2, key_dim=32))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_shape))
    model.compile(optimizer='adam', loss='mse')
    return model

# 构建 注意力 LSTM 模型
def build_LSTM_Attention_model(input_shape, output_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(32, return_sequences=True))
    model.add(AttentionLayer())
    model.add(RepeatVector(input_shape[0]))  # 恢复时间步维度
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_shape))
    model.compile(optimizer='adam', loss='mse')
    return model

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

def build_transformer_model(input_shape, output_shape):
    inputs = Input(shape=input_shape)
    x = Dense(32)(inputs)  # 确保输入形状与 TransformerBlock 的 embed_dim 一致
    x = TransformerBlock(32, num_heads=2, ff_dim=32)(x, training=True)
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(output_shape)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model
def build_parallel_cnn_lstm_model(input_shape, output_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(32, 3, activation='relu')(inputs)
    x = MaxPooling1D(3)(x)
    x = Conv1D(64, 3, activation='relu')(x)
    x = MaxPooling1D(3)(x)
    x = LSTM(32, return_sequences=False)(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(output_shape)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model


def custom_loss(C, lambda1=0.3, lambda2=0.1, lambda3=0.9):
    def loss(y_true, y_pred):
        # 数据损失
        mse_loss = tf.reduce_mean(tf.square(y_true[:, -1] - y_pred))
        #         提取特定特征
        # 由于目标列只包含 mean_rtt，我们不再需要从 y_true 中提取特定特征
        src_ip = y_true[:, 0]  # 假设 src_ip_count 是第一个特征
        print (src_ip)
        mean_pkt = y_true[:, 3]  # 假设 mean_packet_size 是第二个特征

        # 时延-负载损失
        flow_intensity = src_ip * mean_pkt*8
        flow_intensity_mean = tf.reduce_mean(flow_intensity, keepdims=True)  # 计算每个样本的平均流量强度
        rtt_load_loss = tf.reduce_mean(tf.square(y_pred[:, 0] - 0.5 * flow_intensity_mean / C))
        
        # 突发衰减损失（自动微分计算时间导数）
        time = tf.range(tf.shape(y_pred)[0], dtype=tf.float32)  # 确保 time 的形状和类型正确
        time = tf.reshape(time, (-1, 1))
        with tf.GradientTape(watch_accessed_variables=False) as t:
            t.watch(time)
            max_rtt = y_pred  # 确保形状兼容
        d_max_rtt = t.gradient(max_rtt, time)
        
        # 如果 d_max_rtt 为 None，则用 0 替换
        if d_max_rtt is None:
            d_max_rtt = tf.zeros_like(max_rtt)
        
        decay_loss = tf.reduce_mean(tf.square(d_max_rtt + 0.1 * (max_rtt - y_pred)))

        # RTT-窗口损失
        mean_win = y_true[:, 2]
        window_loss = tf.reduce_mean(tf.square(y_pred * mean_win / 100.0))    
        
        # 总损失
        total_loss = mse_loss + lambda1 * decay_loss + lambda3 * window_loss + lambda2 * rtt_load_loss
        return total_loss
    return loss

def build_custom_pinn_lstm(input_shape, C):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(1))  # 输出 mean_rtt
    model.compile(optimizer='adam', loss=custom_loss(C))
    return model

def build_custom_pinn_multi_lstm(input_shape, C):
    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    x = LSTM(32, return_sequences=True)(x)
    attn_output = MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
    x = LSTM(32, return_sequences=False)(attn_output)
    outputs = Dense(1)(x)  # 输出 mean_rtt
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss=custom_loss(C))
    return model
