from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Attention, Concatenate, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def build_attention_model(input_shape):
    inputs = Input(shape=input_shape)

    lstm_out = LSTM(units=64, return_sequences=True)(inputs)
    lstm_out = Dropout(0.2)(lstm_out)

    query = LSTM(units=64, return_sequences=True)(lstm_out)
    attention_out = Attention()([query, query])

    concat = Concatenate()([query, attention_out])
    flatten = Flatten()(concat)

    dense = Dense(units=64, activation='relu')(flatten)
    output = Dense(units=1)(dense)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    return model

