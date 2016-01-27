import os
import pickle
import re
from periscope.naming import class_for_name

def load_from_checkpoint(directory):
    cp = Checkpoint(directory)
    data = (s, e, t, v, classname) = cp.load()
    cls = class_for_name(classname)
    return cls(model=directory, data=data)

class Checkpoint:
    def __init__(self, directory):
        self.directory = directory

    def exists(self):
        return os.path.isdir(self.directory)

    def latest_checkpoint_number(self):
        # scan directory for files
        try:
            files = [n for n in os.listdir(self.directory)
                     if re.match(r'^epoch-\d+\.mdl$', n)]
        except:
            return None
        if len(files) == 0:
            return None
        return max([int(re.match(r'^epoch-(\d+)\.mdl$', x).group(1))
                    for x in files])

    def load(self, epoch=None):
        if epoch is None:
            epoch = self.latest_checkpoint_number()
        if epoch is None:
            raise IOError('no checkpoint found')
        filename = os.path.join(self.directory, 'epoch-%03d.mdl' % epoch)
        with open(filename, 'rb') as f:
            f.seek(0)
            formatver = pickle.load(f)
            state = pickle.load(f)
            epoch = pickle.load(f)
            training = pickle.load(f)
            validation = pickle.load(f)
            if formatver >= 2:
                classname = pickle.load(f)
            return (state, epoch, training, validation, classname)

    def save(self, data):
        os.makedirs(self.directory, exist_ok=True)
        state, epoch, training, validation, classname = data
        filename = os.path.join(self.directory, 'epoch-%03d.mdl' % epoch)
        with open(filename, 'wb+') as f:
            f.seek(0)
            f.truncate()
            formatver = 2
            pickle.dump(formatver, f)
            pickle.dump(state, f)
            pickle.dump(epoch, f)
            pickle.dump(training, f)
            pickle.dump(validation, f)
            pickle.dump(classname, f)
