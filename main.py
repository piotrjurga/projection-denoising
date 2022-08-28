import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
rng = np.random.default_rng(0xcafef00d)

import matplotlib.pyplot as plt

def make_and_train_conv(data):
    train_count = len(data) * 2 // 3
    train_data = data[:train_count]
    test_data = data[train_count:]

    channels = [16, 16, 32, 32, 32]
    encoded_size = 16

    activation = 'leaky_relu'
    inputs = keras.Input(shape=(64, 1), name="projection")
    x = inputs
    for l in channels:
        x = layers.Conv1D(l, 3, activation=activation, padding='same')(x)
        x = layers.MaxPooling1D(2)(x)
        #x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    encoded = layers.Dense(encoded_size, activation=activation)(x)
    first_out_layer_size = 64 // 2**len(channels)
    d = layers.Dense(first_out_layer_size * channels[-1], activation=activation)(encoded)
    d = layers.Reshape((first_out_layer_size, channels[-1]))(d)
    for l in channels[::-1]:
        #d = layers.BatchNormalization()(d)
        d = layers.UpSampling1D(2)(d)
        d = layers.Conv1DTranspose(l, 3, activation=activation, padding='same')(d)
    decoded = layers.Conv1DTranspose(1, 3, activation=activation, padding='same')(d)

    model = keras.Model(inputs, decoded, name="projection_conv")
    model.summary()

    opt = tf.optimizers.Adam(learning_rate=1e-3)
    model.compile(loss='mean_squared_error', optimizer=opt)

    model.fit(train_data, train_data,
            batch_size=64,
            epochs=32,
            validation_data=(test_data, test_data),
            verbose=2)
    model.save('models/conv')

def conv_tests():
    projections = np.fromfile('data/proj16000.dat', dtype=np.float32).reshape([16000,-1])

    projections = projections.reshape([-1, 64, 1])

    #p = projections.copy()
    #p = np.concatenate([p, p[:,::-1]])
    #rng.shuffle(p)
    #make_and_train_conv(p)
    model = tf.keras.models.load_model('models/conv')
    tf.keras.utils.plot_model(model)

    proj_noisy = projections.copy()

    for i, p in enumerate(projections):
        # add noise to the projections
        noise = np.random.normal(0, .002, len(p))
        noise_freq = np.fft.rfft(noise)
        # apply the experimentally determined noise frequency response
        noise_freq *= np.exp(-0.032*np.arange(len(noise_freq)))
        noise = np.fft.irfft(noise_freq)
        noise = noise.reshape((-1, 1))

        proj_noisy[i] += noise

    proj_reco = model(proj_noisy).numpy()

    def filtered(proj):
        freq = np.fft.rfft(proj)
        freq[10:] = 0
        return np.fft.irfft(freq)

    for i in range(3):
        p = projections[i]
        noisy = proj_noisy[i]
        reco = proj_reco[i]

        plt.plot(p.reshape(-1), label='original')
        plt.plot(reco.reshape(-1), label='reconstructed')
        plt.plot(noisy.reshape(-1), label='noisy')
        #plt.plot(filtered(noisy.reshape(-1)), label='noisy filtered')
        plt.legend()
        plt.show()

        plt.plot((np.abs(np.fft.rfft(p.reshape(-1)))), label='original')
        plt.plot((np.abs(np.fft.rfft(reco.reshape(-1)))), label='reconstructed')
        plt.plot((np.abs(np.fft.rfft(noisy.reshape(-1)))), label='noisy')
        plt.legend()
        plt.show()

    proj_noisy.astype(np.float32).tofile('proj_noisy.dat')
    proj_reco.astype(np.float32).tofile('proj_reco.dat')

if __name__ == '__main__':
    conv_tests()
