import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.layers import Flatten, Dense, Input, BatchNormalization, Reshape
from tensorflow.keras.layers import LeakyReLU
import matplotlib.pyplot as plt
import json
from flask import Flask, render_template
import datetime
import os

app = Flask(__name__)


@app.route('/')
def index():
    del_temp()
    return render_template("index.html", message="Predict") 

@app.route('/predict', methods = ['POST', 'GET'])
def predict():
    del_temp()
    latent_dim = 100
    img_shape = (28,28,1)
    model = build_generator(latent_dim, img_shape)
    model.load_weights('saved_models/G.h5')

    filename = 'result{}.png'.format(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    sample_images(model, latent_dim, filename)
    filepath = 'temp/{}'.format(filename)
    return render_template("index.html", message="Retry?", file=filepath)


def del_temp():
    for image in os.listdir('{}/static/temp'.format(os.getcwd())):
        os.remove('{}/static/temp/{}'.format(os.getcwd(), image))

def build_generator(latent_dim, img_shape):

    model = Sequential()

    model.add(Dense(256, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))

    return model


def sample_images(model, latent_dim, filename):
    r, c=5, 5
    noise = np.random.normal(loc=0.0,scale=1.0, size=(r*c,latent_dim))
    gen_imgs = model.predict(noise)

    #rescale images 0 - 1
    gen_imgs = 0.5*gen_imgs + 0.5

    fig, axs = plt.subplots(r,c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='Greys')
            axs[i,j].axis('off')
            cnt+=1

    # filename = 'result{}.png'.format(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    fig.savefig('static/temp/{}'.format(filename))
    plt.close()
    # plt.show()



if __name__ == "__main__":
    app.run()