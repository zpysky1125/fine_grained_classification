import numpy as np
import os
import random
from PIL import Image
from sklearn.model_selection import train_test_split


def to_categorical(labels, num_classes=200):
    return np.eye(num_classes, dtype=np.int64)[np.array(labels).reshape(-1)]


class BirdClassificationGenerator(object):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.num_classes = 200

        self.train_list = []
        self.valid_list = []
        self.test_list = []
        self.train_valid_list = []

        with open(os.path.join(dataset_path, 'splits/train.txt')) as f:
            for line in f.readlines():
                self.train_list.append([line.strip(), int(line.strip().split('.')[0]) - 1])

        with open(os.path.join(dataset_path, 'splits/valid.txt')) as f:
            for line in f.readlines():
                self.valid_list.append([line.strip(), int(line.strip().split('.')[0]) - 1])

        with open(os.path.join(dataset_path, 'splits/test.txt')) as f:
            for line in f.readlines():
                self.test_list.append([line.strip(), int(line.strip().split('.')[0]) - 1])

        self.train_valid_list = self.train_list + self.valid_list


        # with open(os.path.join(dataset_path, 'bounding_boxes_train.txt')) as f:
        #     spamreader = csv.reader(f, delimiter=' ')
        #     for row in spamreader:
        #         self.bb_bird_dict[int(row[0])] = [int(float(x)) for x in row[1:5]]
        #
        # with open(os.path.join(dataset_path, 'image_class_labels_train.txt')) as f:
        #     spamreader = csv.reader(f, delimiter=' ')
        #     for row in spamreader:
        #         self.train_labels_dict[int(row[0])] = int(row[1]) - 1  # offset by 0
        #         # self.train_labels_dict[int(row[0])] = (int(row[1]) -1 )/ 20 #offset by 0
        #
        # with open(os.path.join(dataset_path, 'images_test.txt')) as f:
        #     spamreader = csv.reader(f, delimiter=' ')
        #     for row in spamreader:
        #         self.test_list.append(int(row[0]))
        #         self.images_dict[int(row[0])] = os.path.join('images', row[1])
        #
        # with open(os.path.join(dataset_path, 'bounding_boxes_test.txt')) as f:
        #     spamreader = csv.reader(f, delimiter=' ')
        #     for row in spamreader:
        #         self.bb_bird_dict[int(row[0])] = [int(float(x)) for x in row[1:5]]

    def _shuffle(self):
        random.shuffle(self.train_list)

    def _train_valid_shuffle(self):
        train_list, valid_list = train_test_split(self.train_valid_list, test_size=0.15, shuffle=True)
        self.train_list, self.valid_list = train_list, valid_list

    def _generate(self, idx_list, batch_size):
        loop = 0
        max_size = len(idx_list)
        while True:
            if loop + batch_size < max_size:
                gen_list = idx_list[loop:loop + batch_size]
                loop += batch_size
            else:
                gen_list = idx_list[loop:]
                loop = 0
                self._shuffle()
            assert (len(gen_list) <= batch_size)
            yield zip(*gen_list)

    def _generate_shuffle(self, idx_list, type, batch_size):
        loop = 0
        max_size = len(idx_list)
        while True:
            if loop + batch_size < max_size:
                gen_list = idx_list[loop:loop + batch_size]
                loop += batch_size
            else:
                gen_list = idx_list[loop:]
                loop = 0
                if type == 'train':
                    self._shuffle()
                if type == 'valid':
                    self._train_valid_shuffle()
            assert (len(gen_list) <= batch_size)
            yield zip(*gen_list)

    def train_generator(self, batch_size):
        return self._generate(self.train_list, batch_size)

    def test_generator(self, batch_size):
        return self._generate(self.test_list, batch_size)

    def valid_generator(self, batch_size):
        return self._generate(self.valid_list, batch_size)

    def train_generator_2(self, batch_size):
        return self._generate_shuffle(self.train_list, 'train', batch_size)

    def test_generator_2(self, batch_size):
        return self._generate_shuffle(self.test_list, 'test', batch_size)

    def valid_generator_2(self, batch_size):
        return self._generate_shuffle(self.valid_list, 'valid', batch_size)


# #pre-processing stuff
def gray2rgb(img):
    if len(img.shape) < 3:
        img = np.stack((img,) * 3, axis=2)
    return img


def random_rotation(image, mode=Image.BICUBIC):
    random_angle = np.random.randint(1, 360)
    return image.rotate(random_angle, mode)


def add_random_noise(img):
    return img + np.random.normal(0, 10.0, (img.shape))


def random_flip_lr(img):
    rand_num = np.random.rand(1)
    if rand_num > 0.5:
        img = np.flip(img, 1)
    return img

def random_brightness(img):
    rand_num = np.random.randint(3, high=10, size=1)/10.0
    img = img * rand_num;
    img = img.astype(dtype=np.uint8)
    return img
#
#
# def normalize_input(img, height):
#     img = img.astype(dtype=np.float32)
#     img[:,:,0] -= 103.939
#     img[:,:,1] -= 116.779
#     img[:,:,2] -= 123.68
#     #img = np.divide(img, 255.0)
#     return img
#
#
# def preprocess_image(img, height, width, set_type):
#     img = misc.imresize(np.asarray(img), (height, width))
#     if set_type == 'train':
#         img = random_flip_lr(img)
#         img = random_brightness(img)
#     img = normalize_input(img, height)
#     #img = add_random_noise(img)
#     return img


def get_batch(generator, dataset_path='./CUB_200_2011/CUB_200_2011/images/'):
    imgs = []
    paths, labels = generator.next()
    for i in range(len(paths)):
        img = Image.open(dataset_path + paths[i])
        img = img.resize((224, 224))
        img = np.asarray(img)
        img = gray2rgb(img)
        imgs.append(img)
    imgs = np.asarray(imgs, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    return imgs, labels


def get_batch_2(generator, dataset_path='./CUB_200_2011/CUB_200_2011/images/', type='train'):
    imgs = []
    paths, labels = generator.next()
    for i in range(len(paths)):
        img = Image.open(dataset_path + paths[i])
        img = img.resize((224, 224))
        if type == 'train':
            img = random_rotation(img)
        img = np.asarray(img)
        img = gray2rgb(img)
        if type == 'train':
            img = random_flip_lr(img)
            img = random_brightness(img)
        imgs.append(img)
    imgs = np.asarray(imgs, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    return imgs, labels


if __name__ == '__main__':
    generator = BirdClassificationGenerator("./CUB_200_2011/CUB_200_2011/")
    # generator = cls.valid_generator_2(29)
    # tl, vl = cls.train_list, cls.valid_list
    # print tl[:20]
    # print
    # train_ge = cls.valid_generator(29)
    # for i in range(6):
    #     path, label = train_ge.next()
    #     print label
    #     print len(label)
    # tl, vl = cls.train_list, cls.valid_list
    # print tl[:20]

    # for i in range(33):
    #     path, label = generator.next()
    #     # print label
    #     # print len(label)
    # tl, vl = cls.train_list, cls.valid_list
    # print tl[:20]
    train_generator = generator.train_generator(8)
    get_batch_2(train_generator, './CUB_200_2011/CUB_200_2011/images/', 'train')
