from tensorflow.keras.layers import Layer, Dense, MultiHeadAttention, LayerNormalization, Dropout, Add
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Input
import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow as tf

class MultiHeadAttentionLayer(Layer):
    def __init__(self, num_heads, key_dim, **kwargs):
        super(MultiHeadAttentionLayer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.dense = Dense(key_dim)
        self.additive_attention_dense = Dense(1)

    def call(self, inputs):
        # inputs: (batch_size, seq_len, embed_dim)
        query = inputs[:, -1:, :]  # 最后时间步的隐藏状态 H_D
        key = inputs  # 所有时间步的输出 O
        value = inputs  # 所有时间步的输出 O

        # 点积注意力
        attn_output = self.mha(query, key, value)

        # 加法注意力
        query_expanded = tf.expand_dims(query, axis=2)  # (batch_size, 1, 1, embed_dim)
        key_expanded = tf.expand_dims(key, axis=1)  # (batch_size, 1, seq_len, embed_dim)
        additive_score = K.tanh(self.dense(query_expanded + key_expanded))
        additive_attention_weights = K.softmax(self.additive_attention_dense(additive_score), axis=2)
        additive_attn_output = tf.reduce_sum(additive_attention_weights * key_expanded, axis=2)

        # 合并点积注意力和加法注意力的输出
        combined_attn_output = Add()([attn_output, additive_attn_output])

        return combined_attn_output
class DotProductAttentionLayer(Layer):
    def __init__(self, num_heads, key_dim, **kwargs):
        super(DotProductAttentionLayer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

    def call(self, inputs):
        # inputs: (batch_size, seq_len, embed_dim)
        query = inputs[:, -1:, :]  # 最后时间步的隐藏状态 H_D
        key = inputs  # 所有时间步的输出 O
        value = inputs  # 所有时间步的输出 O

        # 点积注意力
        attn_output = self.mha(query, key, value)

        return attn_output
class AdditiveAttentionLayer(Layer):
    def __init__(self, key_dim, **kwargs):
        super(AdditiveAttentionLayer, self).__init__(**kwargs)
        self.key_dim = key_dim
        self.dense = Dense(key_dim)
        self.additive_attention_dense = Dense(1)

    def call(self, inputs):
        # inputs: (batch_size, seq_len, embed_dim)
        query = inputs[:, -1:, :]  # 最后时间步的隐藏状态 H_D
        key = inputs  # 所有时间步的输出 O
        value = inputs  # 所有时间步的输出 O

        # 加法注意力
        query_expanded = tf.expand_dims(query, axis=2)  # (batch_size, 1, 1, embed_dim)
        key_expanded = tf.expand_dims(key, axis=1)  # (batch_size, 1, seq_len, embed_dim)
        additive_score = K.tanh(self.dense(query_expanded + key_expanded))
        additive_attention_weights = K.softmax(self.additive_attention_dense(additive_score), axis=2)
        additive_attn_output = tf.reduce_sum(additive_attention_weights * key_expanded, axis=2)

        return additive_attn_output
    




def build_custom_pinn_multi_lstm(input_shape, C, lambda1, lambda2, lambda3, attention_type, num_heads, key_dim):
    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    x = LSTM(32, return_sequences=True)(x)
    
    if attention_type == 'add':
        attn_output = AdditiveAttentionLayer(key_dim=key_dim)(x)
    elif attention_type == 'dot':
        attn_output = DotProductAttentionLayer(num_heads=num_heads, key_dim=key_dim)(x)
    elif attention_type == 'multi':
        attn_output = MultiHeadAttentionLayer(num_heads=num_heads, key_dim=key_dim)(x)
    
    x = LSTM(32, return_sequences=False)(attn_output)
    outputs = Dense(1)(x)  # 输出 mean_rtt
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss=custom_loss(C, lambda1, lambda2, lambda3))
    return model

def build_custom_pinn_multi_lstm2(input_shape, C, lambda1, lambda2, lambda3, attention_type, num_heads, key_dim1, key_dim2):
    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inputs)
    if attention_type == 'add':
        attn_output1 = AdditiveAttentionLayer(key_dim=key_dim1)(x)
    elif attention_type == 'dot':
        attn_output1 = DotProductAttentionLayer(num_heads=num_heads, key_dim=key_dim1)(x)
    elif attention_type == 'multi':
        attn_output1 = MultiHeadAttentionLayer(num_heads=num_heads, key_dim=key_dim1)(x)
    x = Dropout(0.3)(x)
    x = LSTM(32, return_sequences=True)(attn_output1)
    if attention_type == 'add':
        attn_output2 = AdditiveAttentionLayer(key_dim=key_dim2)(x)
    elif attention_type == 'dot':
        attn_output2 = DotProductAttentionLayer(num_heads=num_heads, key_dim=key_dim2)(x)
    elif attention_type == 'multi':
        attn_output2 = MultiHeadAttentionLayer(num_heads=num_heads, key_dim=key_dim2)(x)
    x = LSTM(32, return_sequences=False)(attn_output2)
    
    outputs = Dense(1)(x)  # 输出 mean_rtt
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss=custom_loss(C, lambda1, lambda2, lambda3))
    return model

def build_custom_pinn_multi_lstm3(input_shape, output_shape, attention_type, num_heads1,num_heads2, key_dim1, key_dim2):
    model = tf.keras.Sequential()

    # 输入层
    model.add(Input(shape=input_shape))  # 输入形状为 (60, 5)

    # CNN 提取特征
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))  # 输出 (60, 64)
    model.add(tf.keras.layers.Dropout(0.3))  # Dropout 防止过拟合

    # 第一层多头注意力 (dim=64)
    if attention_type == 'add':
        model.add(AdditiveAttentionLayer(key_dim=key_dim1))  # 输出 (60, 64)
    elif attention_type == 'dot':
        model.add(DotProductAttentionLayer(num_heads=num_heads1, key_dim=key_dim1))  # 输出 (60, 64)
    elif attention_type == 'multi':
        model.add(MultiHeadAttentionLayer(num_heads=num_heads1, key_dim=key_dim1))  # 输出 (60, 64)

    # 第一层 LSTM (64 单元)
    model.add(tf.keras.layers.LSTM(64, return_sequences=True))  # 输出 (60, 64)
    model.add(tf.keras.layers.Dropout(0.3))  # Dropout 防止过拟合

    # 第二层多头注意力 (dim=32)
    if attention_type == 'add':
        model.add(AdditiveAttentionLayer(key_dim=key_dim2))  # 输出 (60, 32)
    elif attention_type == 'dot':
        model.add(DotProductAttentionLayer(num_heads=num_heads2, key_dim=key_dim2))  # 输出 (60, 32)
    elif attention_type == 'multi':
        model.add(MultiHeadAttentionLayer(num_heads=num_heads2, key_dim=key_dim2))  # 输出 (60, 32)

    # 第二层 LSTM (32 单元)
    model.add(tf.keras.layers.LSTM(32, return_sequences=False))  # 输出 (batch_size, 32)

    # 恢复时间步维度
    model.add(tf.keras.layers.RepeatVector(input_shape[0]))  # 恢复时间步维度 (60, 32)

    # 第三层 LSTM (32 单元)
    model.add(tf.keras.layers.LSTM(32, return_sequences=False))  # 输出 (batch_size, 32)
    model.add(tf.keras.layers.Dense(32, activation='relu'))  # 全连接层 (32)

    # 输出层
    model.add(tf.keras.layers.Dense(output_shape))  # 输出 (batch_size, output_shape)

    # 编译模型
    model.compile(optimizer='adam', loss='mse')

    return model
def build_custom_pinn_multi_lstm3_layer_contral(input_shape, output_shape, attention_type, num_heads1,num_heads2, key_dim1, key_dim2):
    model = tf.keras.Sequential()

    # 输入层
    model.add(Input(shape=input_shape))  # 输入形状为 (60, 5)

    # CNN 提取特征
    model.add(tf.keras.layers.Conv1D(filters=key_dim1, kernel_size=3, padding='same', activation='relu'))  # 输出 (60, 64)
    model.add(tf.keras.layers.Dropout(0.3))  # Dropout 防止过拟合

    # 第一层多头注意力 (dim=64)
    if attention_type == 'add':
        model.add(AdditiveAttentionLayer(key_dim=key_dim1))  # 输出 (60, 64)
    elif attention_type == 'dot':
        model.add(DotProductAttentionLayer(num_heads=num_heads1, key_dim=key_dim1))  # 输出 (60, 64)
    elif attention_type == 'multi':
        model.add(MultiHeadAttentionLayer(num_heads=num_heads1, key_dim=key_dim1))  # 输出 (60, 64)

    # 第一层 LSTM (64 单元)
    model.add(tf.keras.layers.LSTM(key_dim1, return_sequences=True))  # 输出 (60, 64)
    model.add(tf.keras.layers.Dropout(0.3))  # Dropout 防止过拟合

    # 第二层多头注意力 (dim=32)
    if attention_type == 'add':
        model.add(AdditiveAttentionLayer(key_dim=key_dim2))  # 输出 (60, 32)
    elif attention_type == 'dot':
        model.add(DotProductAttentionLayer(num_heads=num_heads2, key_dim=key_dim2))  # 输出 (60, 32)
    elif attention_type == 'multi':
        model.add(MultiHeadAttentionLayer(num_heads=num_heads2, key_dim=key_dim2))  # 输出 (60, 32)

    # 第二层 LSTM (32 单元)
    model.add(tf.keras.layers.LSTM(key_dim2, return_sequences=False))  # 输出 (batch_size, 32)

    # 恢复时间步维度
    model.add(tf.keras.layers.RepeatVector(input_shape[0]))  # 恢复时间步维度 (60, 32)

    # 第三层 LSTM (32 单元)
    model.add(tf.keras.layers.LSTM(key_dim2, return_sequences=False))  # 输出 (batch_size, 32)
    model.add(tf.keras.layers.Dense(key_dim2, activation='relu'))  # 全连接层 (32)

    # 输出层
    model.add(tf.keras.layers.Dense(output_shape))  # 输出 (batch_size, output_shape)

    # 编译模型
    model.compile(optimizer='adam', loss='mse')

    return model
class PhysicsInformedLossLayer(Layer):
    def __init__(self, feature_scalers, alpha=0.7, beta=0.3, **kwargs):
        super(PhysicsInformedLossLayer, self).__init__(**kwargs)
        self.feature_scalers = feature_scalers
        self.alpha = alpha
        self.beta = beta
    def call(self, inputs):
        y_true, y_pred, features = inputs

        # 确保 y_true 和 y_pred 的形状一致
        if len(y_true.shape) == 2:  # 如果 y_true 是 (batch_size, seq_len)
            y_true = tf.expand_dims(y_true, axis=-1)  # 扩展为 (batch_size, seq_len, 1)

        # 数据拟合损失
        mse_loss = tf.reduce_mean(tf.square(y_pred - y_true))

        # 特征归一化
        src_ip = features[..., 0] / self.feature_scalers['src_ip_count']
        window = features[..., 2] / self.feature_scalers['mean_window_size']
        packet = features[..., 3] / self.feature_scalers['mean_packet_size']

        # 时间导数 ∂RTT/∂t
        dt = 60  # 时间步长 (秒)
        dRTT_dt = (y_pred[:, 1:, :] - y_pred[:, :-1, :]) / dt  # 差分近似时间导数，形状为 (batch_size, seq_len - 1, 1)

        # 截断特征的时间步，使其与 dRTT_dt 的形状一致
        src_ip = src_ip[:, :-1]  # 形状为 (batch_size, seq_len - 1)
        window = window[:, :-1]  # 形状为 (batch_size, seq_len - 1)
        packet = packet[:, :-1]  # 形状为 (batch_size, seq_len - 1)

        # 扩展特征张量的最后一维，使其与 dRTT_dt 的形状一致
        src_ip = tf.expand_dims(src_ip, axis=-1)  # 形状为 (batch_size, seq_len - 1, 1)
        window = tf.expand_dims(window, axis=-1)  # 形状为 (batch_size, seq_len - 1, 1)
        packet = tf.expand_dims(packet, axis=-1)  # 形状为 (batch_size, seq_len - 1, 1)

        # 定义物理方程残差
        residual = dRTT_dt - (
            0.5 * (window / (packet + 1e-6)) -  # 窗口效率项
            0.3 * tf.sqrt(src_ip) +            # 源节点拥塞项
            0.2 * dRTT_dt                      # 时间惯性项
        )
        physics_loss = tf.reduce_mean(tf.square(residual))

        # 总损失
        total_loss = self.alpha * mse_loss + self.beta * physics_loss

        # 返回与 y_true 形状一致的张量
        return tf.ones_like(y_true) * total_loss
