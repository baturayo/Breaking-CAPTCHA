from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
from random import randint
from tqdm import tqdm

def new_captcha(c):
    image = ImageCaptcha(width=280, height=280, font_sizes=[150], fonts=['OpenSans-Light.ttf'])
    ima = image.generate_image(c)
    im = ima.convert('L')
    (width, height) = im.size
    greyscale_map = list(im.getdata())
    greyscale_map = np.array(greyscale_map)
    greyscale_map = greyscale_map.reshape((height, width))

    return greyscale_map


def generate_new_word(l):
    labels = []
    word = ""
    for _ in range(l):
        label = randint(0, 25)
        labels.append(label)
        word += (chr(ord('a')+label))
    return labels, word


def generate_new_pair(l):
    labels, word = generate_new_word(l)
    image = new_captcha(word)
    return image, labels


def next_batch(N=1000, one_hot=True, l=1, indx=0):
    X_s = []
    y_s = []
    if one_hot:
        d = 26
    else:
        d = 1
    for _ in tqdm(range(N)):
        image, labels = generate_new_pair(l)
        X_s.append(image)
        if one_hot:
            # vect = np.zeros(26)
            # vect[labels[indx]] = 1.
            # y_s.append(vect)
            y_s.append(labels)
        else:
            #y_s.append(labels[indx])
            y_s.append(labels)

    return np.array(X_s), np.array(y_s)#.reshape((N, d))


if __name__ == "__main__":
    N_train = 0
    N_test = 0

    X_s_train, y_s_train = next_batch(N=N_train, l=1)
    X_s_test, y_s_test = next_batch(N=N_test, l=1)

    for i in range(0, N_train):
        plt.figure(figsize=(2.8, 2.8))
        plt.imshow(X_s_train[i], cmap="gray")
        plt.axis('off')
        captcha_name = y_s_train[i]
        plt.savefig('captchas/train/%s_%s.jpg' % (captcha_name, i))
        plt.close()

    for i in range(0, N_test):
        plt.imshow(X_s_test[i], cmap="gray")
        plt.axis('off')
        captcha_name = y_s_test[i]
        plt.savefig('captchas/test/%s_%s.jpg' % (captcha_name, i))
        plt.close()

# mask= filters.threshold_otsu(gray)
# plt.imshow(mask, cmap='gray', interpolation='nearest')
# plt.show()
