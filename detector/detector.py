import time

from torch.utils.data import DataLoader, SequentialSampler
from torchvision import datasets, transforms
from facenet_pytorch import InceptionResnetV1
from model.metric import *
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Detector(object):
    def __init__(self, args):
        self.args = args
        self.in_data_dir = args.in_data_dir
        self.out_data_dir = args.out_data_dir
        self.batch_size = args.batch_size
        self.method = args.method
        self.save_dir = args.save_dir

        trans = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
        ])

        in_dataset = datasets.ImageFolder(self.in_data_dir, transform=trans)

        self.in_data_loader = DataLoader(
            in_dataset,
            num_workers=8,
            batch_size=self.batch_size,
            sampler=SequentialSampler(in_dataset)
        )

        out_dataset = datasets.ImageFolder(self.out_data_dir, transform=trans)

        self.out_data_loader = DataLoader(
            out_dataset,
            num_workers=8,
            batch_size=self.batch_size,
            sampler=SequentialSampler(out_dataset)
        )

        self.resnet = InceptionResnetV1(
            classify=True,
            pretrained='vggface2'
        ).to(device)

        self.resnet.eval()

    def detect(self):
        print("Processing in-distribution images")
        self.detect_in_distribution()

        print("Processing out-of-distribution images")
        self.detect_out_distribution()

    def detect_in_distribution(self):
        t0 = time.time()

        f1 = open(os.path.join(self.save_dir, "in_scores.txt"), 'w')
        g1 = open(os.path.join(self.save_dir, "in_labels.txt"), 'w')

        tq_id_test = tqdm(self.in_data_loader)

        with torch.no_grad():
            count = 0

            for inputs, labels in tq_id_test:
                inputs = inputs.to(device)
                labels = labels.to(device)

                curr_batch_size = inputs.shape[0]

                scores = get_msp_score(inputs, self.resnet)

                for score in scores:
                    f1.write("{}\n".format(score))

                outputs = F.softmax(self.resnet(inputs), dim=1)
                outputs = outputs.detach().cpu().numpy()
                preds = np.argmax(outputs, axis=1)
                confs = np.max(outputs, axis=1)

                for k in range(preds.shape[0]):
                    g1.write("{} {} {}\n".format(labels[k], preds[k], confs[k]))

                count += curr_batch_size

                errors = {
                    'Count': count,
                    'Time': time.time() - t0
                }

                tq_id_test.set_postfix(errors)
                t0 = time.time()

    def detect_out_distribution(self):
        f2 = open(os.path.join(self.save_dir, "{}_out_scores.txt".format(self.method)), 'w')

        tq_out_test = tqdm(self.out_data_loader)

        t0 = time.time()

        count = 0

        for images, labels in tq_out_test:
            images = images.cuda()
            labels = labels.cuda()

            curr_batch_size = images.shape[0]

            scores = self.get_score(images)

            for score in scores:
                f2.write("{}\n".format(score))

            count += curr_batch_size

            errors = {
                'Count': count,
                'Time': time.time() - t0
            }

            tq_out_test.set_postfix(errors)
            t0 = time.time()

    def get_score(self, inputs, T=0.1):
        if self.method == "msp":
            scores = get_msp_score(inputs, self.resnet)

        elif self.method == "energy":
            scores = get_energy_score(inputs, self.resnet, T)

        elif self.method == "odin":
            scores = get_odin_score(inputs, self.resnet)

        else:
            assert False, 'Not supported method'

        return scores
