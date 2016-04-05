import numpy as np
import os
from PIL import Image, ImageOps

"""
A corpus is represented by a directory containing the following files:

dimensions.txt     - "h w" dimensions of the images - all the same shape.
labelnames.txt     - human-readable names of all the labels
train.names.txt    - human-readable filenames for the training set images
train.images.db    - int8 RGB training set images
train.labels.db    - int32 labels for training set images
val.names.txt      - human-readable filenames for validation set images
val.images.db      - int8 RGB validation set images
val.labels.db      - int32 labels for validation set images

prepare_corpus creates a directory with these files by consuming
various other files.
"""

class Corpus:
    """
    Corpus(directory) to load a corpus from a directory

    corpus.get('train', shape=(h,w), batch_size=256, randomize=True)
       this returns a training set iterable.  each item in the iterable
       is a tuple (imagetensor, labelvector, namelist), representing a
       batch of cases, with images cropped to the given shape, and
       batches limited to the given batch size.  If randomize is set,
       the batches are not provided in sequential order, and the
       images are randomly cropped and flipped.
    """
    def __init__(self, directory):
        self.directory = directory
        self.load_dimensions()
        self.load_labelnames()
        self.names = {}
        self.X = {}
        self.Y = {}
        for kind in ['train', 'val']:
            self.names[kind] = self.load_imagenames('%s.names.txt' % kind)
            self.Y[kind] = self.map_labels('%s.labels.db' % kind)
            self.X[kind] = self.map_images('%s.images.db' % kind)
            assert len(self.X[kind]) == len(self.names[kind])
            assert len(self.Y[kind]) == len(self.names[kind])

    def get(self, kind=None, index=0, shape=None, randomize=False):
        """
        Gets a single image as a (image, label, name) triple, without
        batching.
        """
        x = self.X[kind][index,:,:,:]
        y = self.Y[kind][index]
        name = self.names[kind][index]
        if shape is None:
            shape = x.shape[1:]
        cropv = x.shape[1] - shape[0]
        croph = x.shape[2] - shape[1]
        if cropv > 0 or croph > 0:
            if randomize:
                # random horizontal flip
                if np.random.randint(2):
                    x = x[:,:,::-1]
                # randomize crop
                top = np.random.randint(cropv + 1)
                left = np.random.randint(croph + 1)
            else:
                # center crop
                top = cropv // 2
                left = croph // 2
            x = x[:,left:left+shape[0],top:top+shape[1]]
        return (x, y, name)

    def batches(self,
            kind=None,         # 'train' or 'val'
            batch_size=256,    # Number of input images in a batch
            shape=None,        # Cropping shape.
            randomize=False,   # Whether to flip+randomly translate crop.
            limit=None):       # Limit the corpus to this length/slice/subset.
        """
        Returns an iterable that returns (input, label, filename) batches.
        """
        X = self.X[kind]
        Y = self.Y[kind]
        names = self.names[kind]
        if limit is not None:
            if isinstance(limit, int):
                limit = slice(0, limit)
            X = X[limit]
            Y = Y[limit]
            names = names[limit]
        if shape is None:
            shape = X.shape[2:]
        return CorpusIterable(X, Y, names, batch_size, shape, randomize)

    def load_dimensions(self):
        """
        Loads dimensions.txt file, which should be in the format
        '128 192' for dimension of height 128 and width 192.
        """
        with open(os.path.join(self.directory, 'dimensions.txt'), 'r') as f:
            txt = f.read()
            self.dimensions = tuple([int(n) for n in txt.split()])
            assert len(self.dimensions) == 2

    def load_labelnames(self):
        """
        Loads labelnames.txt file, which should list one human-readable
        label name per line.
        """
        with open(os.path.join(self.directory, 'labelnames.txt'), 'r') as f:
            txt = f.read()
            self.labelnames = tuple(
                [n.split(None, 1)[0] for n in txt.split('\n') if n.strip()])
            self.labelcount = len(self.labelnames)

    def load_imagenames(self, filename):
        """
        Loads labelnames.txt file, which should list one human-readable
        label name per line.
        """
        with open(os.path.join(self.directory, filename), 'r') as f:
            txt = f.read()
            return tuple(
                [n.split(None, 1)[0] for n in txt.split('\n') if n.strip()])

    def map_labels(self, filename):
        """
        Loads the labels for a memory-map_ped labelled image set, organized
        as a numpy array of int32, one per case.
        """
        return np.memmap(
            os.path.join(self.directory, filename), dtype=np.int32, mode='r')

    def map_images(self, filename):
        """
        Loads the images for a memory-map_ped image set, organized
        as a numpy array of int8 in (case, rgb, y, x) order.
        """
        a = np.memmap(
            os.path.join(self.directory, filename), mode='r',
            dtype=np.int8)
        cases = len(a) // 3 // self.dimensions[0] // self.dimensions[1]
        a.shape = (cases, 3, self.dimensions[0], self.dimensions[1])
        return a


class CorpusIterable:
    def __init__(self, X, Y, names, batch_size, shape, randomize):
        self.X = X
        self.Y = Y
        self.names = names
        self.batch_size = batch_size
        self.shape = shape
        self.randomize = randomize

    def __len__(self):
        return len(self.X) // self.batch_size + (
            len(self.X) % self.batch_size > 0)

    def count(self):
        return len(self.X)

    def __iter__(self):
        end = len(self.X)
        cropv = self.X.shape[2] - self.shape[0]
        croph = self.X.shape[3] - self.shape[1]
        if self.randomize:
            start = np.random.randint(end)
            steps = [n % end for n in range(start, end + start, self.batch_size)]
            np.random.shuffle(steps)
        else:
            steps = range(0, end, self.batch_size)
            # center crop
            top = cropv // 2
            left = croph // 2
        for start_idx in steps:
            if self.randomize and start_idx + self.batch_size > end:
                # Handle wraparound case
                e1 = slice(start_idx, end)
                e2 = slice(0, (start_idx + self.batch_size) % end)
                x = np.concatenate([self.X[e1], self.X[e2]])
                y = np.concatenate([self.Y[e1], self.Y[e2]])
                n = self.names[e1] + self.names[e2]
            else:
                excerpt = slice(start_idx, start_idx + self.batch_size)
                x = self.X[excerpt]
                y = self.Y[excerpt]
                n = self.names[excerpt]
            if self.randomize:
                # random horizontal flip
                if np.random.randint(2):
                    x = x[:,:,:,::-1]
                # randomize crop
                top = np.random.randint(cropv + 1)
                left = np.random.randint(croph + 1)
            x = x[:,:,left:left+self.shape[0],top:top+self.shape[1]]
            yield (x, y, n)


def prepare_corpus(devkit_dir, data_dir, target_dir, width, height,
        seed=None, pretty=None):
    """
    Creates corpus files by preprocessing a directory of images.
    Input:
       (devkit-directory)/categories.txt "name index"
       (devkit-directory)/train.txt      "path labelnum"
       (devkit-directory)/val.txt        "path labelnum"
       (data-directory)/train/...
       (data-directory)/val/...
    Output:
       (target-directory)/.... (as described above)
    """
    creator = CorpusCreator(devkit_dir, data_dir, target_dir)
    creator.prepare(width, height, seed, pretty)

class CorpusCreator:
    def __init__(self, devkit_dir, data_dir, target_dir):
        self.devkit = devkit_dir
        self.data = data_dir
        self.target = target_dir

    def prepare(self, width, height, seed=None, pretty=None):
        if pretty:
            pretty.task('Creating corpus from {} and {} into {}x{} {}'.format(
                self.devkit, self.data, width, height, self.target))
        os.makedirs(self.target, exist_ok=True)
        if seed is not None:
           self.write_seed_file(seed)
        self.write_dimensions(width, height)
        self.write_label_names()
        self.process_images('train', width, height, seed=seed, pretty=pretty)
        self.process_images('val', width, height, seed=(seed + 1000),
            link=True, texture=2048, pretty=pretty)

    def load_devkit_file(self, filename):
        """
        Loads a file that has "name num" rows, returning a list of
        (name num) tuples.
        """
        with open(os.path.join(self.devkit, filename), 'r') as f:
            txt = f.read()
            strings = [n.split(None, 2)[:2] for n in txt.split('\n') if n]
            return [(n, int(i)) for n, i in strings]

    def write_seed_file(self, seed):
        with open(os.path.join(self.target, "seed.txt"), 'w') as f:
            f.write('{}\n'.format(seed))

    def write_dimensions(self, w, h):
        with open(os.path.join(self.target, "dimensions.txt"), 'w') as f:
            f.write('{} {}\n'.format(h, w))

    def write_label_names(self):
        labelnames = self.load_devkit_file('categories.txt')
        labelarray = [None] * len(labelnames)
        for n, i in labelnames:
            labelarray[i] = n
        with open(os.path.join(self.target, "labelnames.txt"), 'w') as f:
            for i in range(len(labelarray)):
                f.write('{} {}\n'.format(labelarray[i], i))

    def write_image_names(self, imagenames, filename):
        with open(os.path.join(self.target, filename), 'w') as f:
            for i, (name, label) in enumerate(imagenames):
                f.write('{} {} {}\n'.format(name, label, i))

    def load_pil_image(self, filename, width, height):
        im = Image.open(os.path.join(self.data, filename))
        im.load()
        if (width, height) != im.size:
            im = ImageOps.fit(im, (width, height), Image.ANTIALIAS)
        return im

    def numpy_image(self, im):
        npa = (np.array(im.convert('RGB')) - 128).astype(np.int8)
        return npa.transpose((2, 0, 1))

    def link_image(self, filename, kind, index):
        source = os.path.abspath(os.path.join(self.data, filename))
        target = os.path.join(self.target, 'link', kind, 'i%d.jpg' % index)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        try:
            os.remove(target)
        except OSError:
            pass
        os.symlink(source, target)

    def process_images(self, prefix, width, height,
             seed=None, pretty=None, link=False, texture=0):
        if pretty:
            pretty.subtask('Processing {} images'.format(prefix))
        imagenames = self.load_devkit_file('{}.txt'.format(prefix))
        np.random.RandomState(seed=seed).shuffle(imagenames)
        self.write_image_names(imagenames, '{}.names.txt'.format(prefix))
        la = np.memmap(
            os.path.join(self.target, '{}.labels.db'.format(prefix)),
            shape=len(imagenames), dtype=np.int32, mode='w+')
        ia = np.memmap(
            os.path.join(self.target, '{}.images.db'.format(prefix)),
            shape=(len(imagenames), 3, height, width), dtype=np.int8, mode='w+')
        if texture:
            si = None
            sw = texture // width
            sh = texture // height
            sn = 0
        if pretty:
            p = pretty.progress(len(imagenames))
        for i, (name, label) in enumerate(imagenames):
            if pretty:
                p.update(i)
            la[i] = label
            im = self.load_pil_image(name, width, height)
            ia[i,:,:,:] = self.numpy_image(im)
            if texture:
                sx = (i % sw) * width
                sy = ((i // sw) % sh) * height
                if (not sx and not sy) or i == (len(imagenames) - 1):
                    if si is not None:
                        si.save(
                            os.path.join(self.target, '{}.{}.tex.jpg'.format(
                                prefix, sn)),
                            "JPEG",
                            subsampling=0,
                            quality=95)
                        sn += 1
                    si = Image.new('RGB', (texture, texture))
                si.paste(im, (sx, sy))
            if link:
                self.link_image(name, prefix, i)
        if pretty:
            p.finish()
