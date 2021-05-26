import matplotlib.pyplot as plt
import cv2
import albumentations as A
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array


def albument(image, quantity=12):

    # Declare an augmentation pipeline
    transform = A.Compose([
        A.HorizontalFlip(p=0.3),
        A.RGBShift(p=0.3),
        A.ChannelShuffle(p=0.2),
        A.RandomBrightnessContrast(p=0.2),
        A.ElasticTransform(p=0.3),
        A.GaussianBlur(p=0.3),
        A.ToGray(p=0.2),
        A.CLAHE(p=0.3),
    ])

    # Augment an image
    for i in range(quantity):
        transformed = transform(image=image)
        t_image = transformed["image"]

        t_image = cv2.cvtColor(t_image, cv2.COLOR_BGR2RGB)

        cv2.imwrite(f'data/albumentation/isaque_{i}.png', t_image)


def keras_algument(image, quantity=12):

    # Declare augmentation parameters
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2,
        zoom_range=0.3,
        channel_shift_range=0.3
    )

    x = img_to_array(image)
    x = x.reshape((1,) + x.shape)

    i = 0
    for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='data/keras', save_prefix='isaque', save_format='png'):
        i += 1
        if i == quantity:
            break


def main():
    
    # Read an image with OpenCV and convert it to the RGB colorspace
    image = cv2.imread("data/isaque.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(image.shape)

    quantity = 12

    albument(image, quantity)
    keras_algument(image, quantity)


if __name__ == '__main__':
    main()