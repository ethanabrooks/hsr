import dill as pickle

class Hyperparameters(object):

    def __init__(self):
        self.uninitialized = True


    def init(self, path, restore, force_reinitialization=False):
        if not self.uninitialized and not force_reinitialization:
            raise Exception('Reinitialization of Hyperparameters singleton.')

        self.path = path
        if not restore:
            self.hyperparam_dict = {}
        else:
            with open(path, 'rb') as f:
                self.hyperparam_dict = pickle.load(f)

    def __call__(self, name, value):
        if name in self.hyperparam_dict:
            return self.hyperparam_dict[name]
        else:
            self.hyperparam_dict[name] = value
            self.dump_file()
            return value

    def dump_file(self):
        with open(self.path, 'wb') as f:
            pickle.dump(self.hyperparam_dict, f)


HYP = Hyperparameters()


