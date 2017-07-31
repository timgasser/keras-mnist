from collections import namedtuple
import pickle

from sklearn.model_selection import train_test_split

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

    def load_data(self):
        """
        Loads data from the given pickle files stored in local variables
        """   
        if self.verbose:
            print('Loading pickle files')

        try:
            X_all = pickle.load(open(self.directory + self.X_train_file, "rb"))
            y_all = pickle.load(open(self.directory + self.y_train_file, "rb"))
            
            
            X_test = pickle.load(open(self.directory + self.X_test_file, "rb"))
            y_test = pickle.load(open(self.directory + self.y_test_file, "rb"))
        
            X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=5000)
        
        except Exception as e:
            print('Error loading pickle file: {}'.format(e))
            return None
        
        # Return a namedtuple with the data
        DataSet = namedtuple('dataset', ['train', 'val', 'test'])

        self.X = DataSet(train=X_train, val=X_val, test=X_test)
        self.y = DataSet(train=y_train, val=y_val, test=y_test)

        return self.X, self.y






