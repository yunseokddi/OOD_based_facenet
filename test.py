import torch

from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler
from torchvision import datasets, transforms
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training, extract_face
from model.metric import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

if __name__ == "__main__":
    data_dir = '/home/dorosee/yunseok/data/lfw/lfw'
    pairs_path = '/home/dorosee/yunseok/data/lfw/pairs.txt'

    batch_size = 16
    epochs = 15
    workers = 0 if os.name == 'nt' else 8

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    mtcnn = MTCNN(
        image_size=160,
        margin=14,
        device=device,
        selection_method='center_weighted_size'
    )

    # Define the data loader for the input set of images
    orig_img_ds = datasets.ImageFolder(data_dir, transform=None)

    # overwrites class labels in dataset with path so path can be used for saving output in mtcnn batches
    orig_img_ds.samples = [
        (p, p)
        for p, _ in orig_img_ds.samples
    ]

    loader = DataLoader(
        orig_img_ds,
        num_workers=workers,
        batch_size=batch_size,
        collate_fn=training.collate_pil
    )

    crop_paths = []
    # box_probs = []
    #
    for i, (x, b_paths) in enumerate(loader):
        crops = [p.replace(data_dir, data_dir + '_cropped') for p in b_paths]
    #     mtcnn(x, save_path=crops)
        crop_paths.extend(crops)
    #     print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')
    #
    # # Remove mtcnn to reduce GPU memory usage
    # del mtcnn
    # torch.cuda.empty_cache()

    # create dataset and data loaders from cropped images output from MTCNN

    trans = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])

    dataset = datasets.ImageFolder(data_dir + '_cropped', transform=trans)

    embed_loader = DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset)
    )

    # Load pretrained resnet model
    resnet = InceptionResnetV1(
        classify=False,
        pretrained='vggface2'
    ).to(device)

    classes = []
    embeddings = []
    resnet.eval()
    with torch.no_grad():
        for xb, yb in embed_loader:
            xb = xb.to(device)
            b_embeddings = resnet(xb)
            b_embeddings = b_embeddings.to('cpu').numpy()
            classes.extend(yb.numpy())
            embeddings.extend(b_embeddings)

    # print(classes)
    embeddings_dict = dict(zip(crop_paths, embeddings))

    pairs = read_pairs(pairs_path)
    print(pairs)
    path_list, issame_list = get_paths(data_dir + '_cropped', pairs)
    embeddings = np.array([embeddings_dict[path] for path in path_list])

    # print(embeddings)
    # print(issame_list)

    tpr, fpr, accuracy, val, val_std, far, fp, fn = evaluate(embeddings, issame_list)

    print(np.mean(accuracy))
