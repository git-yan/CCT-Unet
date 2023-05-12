from torchvision import transforms
from dataset import ProstateDataset_Test
from augmentations import JointTransform
from torch.utils.data import DataLoader
import cv2
from CCT_Unet import CCT_Unet
from CSWin_Unet import CSWin_Unet
from one_hot import onehot_to_mask


def test_patient(args, model, device):
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    tf_test = JointTransform(crop_size=(224, 224), resize_After=(224, 224))
    test_datasets = ProstateDataset_Test(args.test_dataset, tf_test, one_hot_mask=args.one_hot)
    test_dataloaders = DataLoader(test_datasets, batch_size=1, num_workers=0, worker_init_fn=worker_init_fn)

    with torch.no_grad():
        model.load_state_dict(torch.load(args.load))
        model.eval()

    palette = [[0], [127], [255]]
    for imgs, anns, cor in test_dataloaders:
        pred = torch.sigmoid(model(imgs.cuda()))
        pred[pred < 0.5] = 0
        pred[pred > 0.5] = 1
        batch_size = anns.shape[0]
        for j in range(batch_size):
            pred = pred.detach().cpu().numpy()[j].transpose([1, 2, 0])
            pred = onehot_to_mask(pred, palette).transpose([2, 0, 1])

        pred_last = pred[0]
        img = np.array(transforms.ToPILImage()(pred_last))  # Convert a tensor or an ndarray to PIL Image.
        pad_size = int((cor[0] - 224) / 2)
        img = cv2.copyMakeBorder(img, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=[0])
        img = transforms.ToPILImage()(img)  # Convert a tensor or an ndarray to PIL Image.
        img.save('test_IMG.png')


if __name__ == "__main__":
    import argparse
    import os
    import random
    import torch
    import numpy as np
    import random
    import torch
    import torch.backends.cudnn as cudnn

    parser = argparse.ArgumentParser(description='Prostate')
    parser.add_argument('--numclass', default=3, type=int)
    parser.add_argument('--one_hot', default=True, type=bool)
    parser.add_argument('--load', default=r'model_check/CCT-Unet.pth', type=str)
    parser.add_argument('--imgsize', type=int, default=224)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--cuda', default="on", type=str, help='switch on/off cuda option (default: off)')
    parser.add_argument('--CUDA_device', default="1", type=str)
    parser.add_argument('--n_gpu', type=int, default=2, help='total gpu')
    parser.add_argument('--deterministic', type=bool, default=True, help='whether use deterministic training')
    parser.add_argument('--test_dataset', default=r'data', type=str)

    args = parser.parse_args()
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = False

    model = CCT_Unet(in_chans=3, num_classes=3, patch_size=4, embed_dim=64, depth=[1, 2, 21, 1],
                     split_size=[1, 2, 7, 7], num_heads=[2, 4, 8, 16], mlp_ratio=4.)
    device = torch.device("cuda")
    if args.n_gpu > 1:
        print("let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    else:
        print("please s use 2 GPU!")
        os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_device
    model = model.to(device)
    test_patient(args, model, device)
