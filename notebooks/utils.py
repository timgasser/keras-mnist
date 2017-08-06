import numpy as np

from collections import namedtuple
import pickle

from sklearn.preprocessing import LabelBinarizer

# Relevant layers in keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Activation, AveragePooling2D, Flatten


# Utility functions for the MNIST notebook

class MnistDataSet(object):
    """
    Wrapper class to load and process the MNIST data as needed
    """

    def __init__(self, directory, X_train_file='train-images-idx3-ubyte.pkl', 
                                  y_train_file='train-labels-idx1-ubyte.pkl',
                                  X_test_file='t10k-images-idx3-ubyte.pkl', 
                                  y_test_file='t10k-labels-idx1-ubyte.pkl',
                                  random=1234,
                                  verbose=False):
        """
        Creates a new instance of the class, with the directories of
        the processed pickle files 
        """
        self.directory = directory
        self.X_train_file = X_train_file
        self.y_train_file = y_train_file
        self.X_test_file = X_test_file
        self.y_test_file = y_test_file
        self.random = random # Use for reproducible results
        self.verbose = verbose

    def data(self):
        """
        Loads data and performs all processing before returning it
        """
        mnist = MnistDataSet(directory='../input/')
        X_train, y_train, X_test, y_test = self.load_data()
        y_train, y_test = self.onehot_encode_labels(y_train, y_test)
        X_train, X_test = self.add_image_borders(X_train, 2, 0), mnist.add_image_borders(X_test, 2, 0)
        X_train, X_test = self.add_image_dim(X_train), mnist.add_image_dim(X_test)
        return X_train, y_train, X_test, y_test

    def load_data(self):
        """
        Loads data from the given pickle files stored in local variables
        """   
        
        if self.verbose:
            print('Loading pickle files')

        try:
            X_train = pickle.load(open(self.directory + self.X_train_file, "rb"))
            y_train = pickle.load(open(self.directory + self.y_train_file, "rb"))
            
            X_test = pickle.load(open(self.directory + self.X_test_file, "rb"))
            y_test = pickle.load(open(self.directory + self.y_test_file, "rb"))
        
        except Exception as e:
            print('Error loading pickle file: {}'.format(e))
            return None

        return X_train, y_train, X_test, y_test

    def onehot_encode_labels(self, y_train, y_test):
        """
        Converts a 1-d array of values into a 2-d onehot array
        """ 
        lbe = LabelBinarizer()
        lbe.fit(y_train)
        y_train_ohe = lbe.transform(y_train)
        y_test_ohe = lbe.transform(y_test)

        return y_train_ohe, y_test_ohe

    def add_image_dim(self, images, after=True):
        """
        Adds an extra dimension to monochrome image files, optionally before
        the X and Y dimensions
        """

        if after:
            new_images = images[:,:,:, np.newaxis]
        else:
            new_images = images[np.newaxis:,:,:]
        
        return new_images

    def image_border(self, image, size, fill):
        """
        Adds a border around the nupmy array of the gizen size and value
        """
        im_w, im_h = image.shape
        im_dtype = image.dtype
        
        new_image = np.full((im_w + (2 * size), im_h + (2 * size)),
                            fill_value=fill, dtype=im_dtype)
        new_image[size:im_h + size, size:im_w + size] = image
        
        assert new_image.dtype == image.dtype
        assert new_image.shape[0] == image.shape[0] + (2 * size)
        assert new_image.shape[1] == image.shape[1] + (2 * size)
        assert np.array_equal(image, new_image[size:size+im_h, size:size+im_w])
        return new_image

    def add_image_borders(self, images, size, fill):
        """
        Adds image borders to an array of images
        """
        new_images = np.zeros((images.shape[0], 
                               images.shape[1] + (2 * size),
                               images.shape[2] + (2 * size)),
                               dtype = images.dtype)

        for idx in range(images.shape[0]):
            new_images[idx] = self.image_border(images[idx], 2, 0)
        return new_images

                
def lenet5_model(verbose=False):
    """
    Creates and returns a lenet5 model
    """

    # Create the model 
    model = Sequential()

    model.add(Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), input_shape=(32, 32, 1))) # C1
    model.add(AveragePooling2D(pool_size=(2, 2))) # S2
    model.add(Activation('tanh'))

    model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1))) # C3
    model.add(AveragePooling2D(pool_size=(2, 2))) # S4
    model.add(Activation('tanh'))

    model.add(Conv2D(filters=120, kernel_size=(5, 5), strides=(1, 1))) # C5
    model.add(Activation('tanh'))

    model.add(Flatten())
    model.add(Dense(120)) # F6
    model.add(Activation('tanh'))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    if verbose:
        print(model.summary())
    return model

# Net training methods
class ModelEvaluator(object):
    """
    Singleton class used to train and evaluate models. All the results are
    stored for later comparison
    """

    def __init__(self):
        self.models = dict() 
        self.results = dict()
        self.history = dict()

    def evaluate_model(self, tag, 
                       model, optimizer, 
                       X_train, y_train, 
                       X_test, y_test,
                       batch_size, epochs, 
                       verbose=False):
        """
        Wrapper method to create, train and optionally CV, and check performance on test set
        """

        def lr_value(self, epoch):
            """
            Returns the learning rate based on the epoch
            """
            if epoch <= 2:
                return 0.0005
            elif epoch <= 5:
                return 0.0002
            elif epoch <= 

        print('Compiling model')
        if verbose:
            model.summary()
            
        model.compile(optimizer=optimizer,
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])

        np.random.seed(1234)

        print('Training model') 
        history = model.fit(X_train, y_train, 
                            validation_data=(X_test, y_test),
                            batch_size=batch_size,
                            epochs=epochs, 
                            callbacks=[lr_scheduler],
                            verbose=1 if verbose else 0)

        print('Evaluating model')
        score = model.evaluate(X_test, y_test, batch_size=batch_size)
        test_error = 1.0 - score[1]

        print('Test error: {:.4f}'.format(test_error))

        if verbose:
            print('Test results: Loss = {:.4f}, Error = {:.4f}'.format(score[0], test_error))
                
        self.models[tag] = model
        self.results[tag] = test_error
        self.history[tag] = history.history

