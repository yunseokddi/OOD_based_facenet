import os

from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image


def main(images_path):
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    resnet.classify = True

    for file in os.listdir(images_path):
        img_path = os.path.join(images_path, file)
        img = Image.open(img_path)

        img_probs = resnet(img.unsqueeze(0))

        print(img_probs)


if __name__ == "__main__":
    images_path = "/home/dorosee/yunseok/data/lfw-samples"

    main(images_path)
