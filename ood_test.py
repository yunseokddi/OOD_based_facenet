import argparse

from model.metric import *
from detector.detector import Detector


parser = argparse.ArgumentParser(description="OOD Based Face Detector")

parser.add_argument('--in-data-dir', default='/home/dorosee/yunseok/data/VGGface_2/', type=str)
parser.add_argument('--out-data-dir', default='/home/dorosee/yunseok/data/Korean_face/OOD_test/', type=str)
parser.add_argument('--batch-size', default=256, type=int)
parser.add_argument('--method', default='odin', type=str)
parser.add_argument('--save-dir', default='./result/', type=str)

parser.set_defaults(argument=True)

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main(args):
    detector = Detector(args)
    detector.detect()

    compute_traditional_odd(args.method)

if __name__ == "__main__":
    main(args)