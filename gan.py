#MNIST generator
import numpy as np
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Input, BatchNormalization, Reshape
from keras.layers.advanced_activations import LeakyReLU
import matplotlib.pyplot as plt

class GAN():
    def __init__(self):
        self.img_rows=28
        self.img_columns=28
        self.channels=1
        self.img_shape=(self.img_rows, self.img_columns, self.channels)
        self.latent_dim=100

        optimizer=Adam(lr=0.0002, beta_1=0.5)

        #Build and compile discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        #Build generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)
        self.combined=self.gan(G=self.generator, D=self.discriminator)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        #model.summary()

        return model

    def build_discriminator(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        #model.summary()

        return model

    def gan(self, G, D):
        model = Sequential()
        model.add(G)
        D.trainable=False
        model.add(D)
        return model


    def train(self, epochs, batch_size, sampling_interval):
        print("loading data.......")
        (X_train,_),(_,_) = mnist.load_data()

        #rescale dataset
        X_train = X_train.astype(np.float32)/127.5 - 1.
        X_train = np.expand_dims(X_train,axis=3)
        #print(X_train.shape)

        #labels
        valid = np.ones((batch_size,1), dtype=np.float32)
        fake = np.zeros((batch_size,1), dtype=np.float32)

        for epoch in range(epochs):
            #------------
            #Train D
            #------------

            #select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            #print(imgs.shape)
            #print('debug!')

            noise = np.random.normal(loc=0.0, scale=1.0, size=(batch_size, self.latent_dim))
            noise = noise.astype(np.float32)
            #print(noise.shape)
            #print('debug!')
            #generate a batch of new images
            gen_imgs = self.generator.predict(noise)
            #print('debug!')
            #train discriminator
            d_loss_real=self.discriminator.train_on_batch(imgs,valid)
            d_loss_fake=self.discriminator.train_on_batch(gen_imgs,fake)
            d_loss = np.add(d_loss_real,d_loss_fake)

            #print('debug!')
            #-------------
            #Train G
            #-------------

            noise = np.random.normal(loc=0.0, scale=1.0, size=(batch_size, self.latent_dim))
            #Train generator to fool discriminator
            g_loss = self.combined.train_on_batch(noise,valid)

            #plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]"%(epoch, d_loss[0], 100*d_loss[1], g_loss))

            #save generated images from time to time
            if epoch%sampling_interval == 0:
                self.sample_images(epoch)
                self.saver()
        self.saver()
    def savr(self):
        #saving generator to disk
        G_json = self.generator.to_json()
        with open("saved_models/G.json", "w") as json_file:
            json_file.write(G_json)
        # serialize weights to HDF5
        self.generator.save_weights("saved_models/G.h5")
        print("Saved G to disk")
        #saving discriminator to disk
        D_json = self.discriminator.to_json()
        with open("saved_models/D.json", "w") as json_file:
            json_file.write(D_json)
        # serialize weights to HDF5
        self.discriminator.save_weights("saved_models/D.h5")
        print("Saved D to disk")



    def sample_images(self, epoch):
        r, c=5, 5
        noise = np.random.normal(loc=0.0,scale=1.0, size=(r*c,self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        #rescale images 0 - 1
        gen_imgs = 0.5*gen_imgs + 0.5

        fig, axs = plt.subplots(r,c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='Greys')
                axs[i,j].axis('off')
                cnt+=1
        fig.savefig("images/%d.png"%epoch)
        plt.close()



def main():
    gan = GAN()
    gan.train(epochs=30000, batch_size=32, sampling_interval=200)

if __name__ == '__main__':
    main()
