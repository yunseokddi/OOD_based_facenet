import torch
import torch.nn as nn
import argparse

from torch.utils.data import DataLoader, SequentialSampler
from torchvision import datasets, transforms
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from model.metric import *

parser = argparse.ArgumentParser(description="OOD Based Face Detector")

parser.add_argument('--data-dir', default='/home/dorosee/yunseok/data/VGGface_2/', type=str)
parser.add_argument('--batch-size', default=16, type=int)

parser.set_defaults(argument=True)

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main(data_dir, batch_size):
    trans = transforms.Compose([
        transforms.Resize(160),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(data_dir, transform=trans)

    embed_loader = DataLoader(
        dataset,
        num_workers=4,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset)
    )

    resnet = InceptionResnetV1(
        classify=True,
        pretrained='vggface2'
    ).to(device)

    softmax = nn.Softmax(dim=1)
    resnet.eval()

    with torch.no_grad():
        for idx, (xb, yb) in enumerate(embed_loader):
            print(xb)
            xb = xb.to(device)
            b_embeddings = resnet(xb)
            # -----------------------------------------------
            softmax_result = softmax(b_embeddings)
            softmax_result = softmax_result.to('cpu').numpy()

            print(np.sum(softmax_result, axis=1))

            b_embeddings = b_embeddings.to('cpu').numpy()
            # print(b_embeddings.shape)
            # print(b_embeddings)
            break
            classes.extend(yb.numpy())
            embeddings.extend(b_embeddings)


if __name__ == "__main__":
    main(args.data_dir, args.batch_size)
