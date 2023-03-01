import torch
from torch.utils.data import DataLoader
import odak
from lensless.models.simple_unet import UNet
from lensless.helpers.diffusercam import DiffuserCam
from tqdm import tqdm

DIFFUSERCAM_DIR = "/cs/student/projects1/2020/sonanuga/dataset"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate(testing_loader, model, device=DEVICE):
    for i, data in enumerate(tqdm(testing_loader, 0)):
        input, lensed = data
        input = input.to(device)
        lensed = lensed.to(device)
        output = model(input)

        image = torch.permute(torch.squeeze(output, 0), (2, 1, 0))
        label = torch.permute(torch.squeeze(lensed), (2, 1, 0))

        odak.learn.tools.save_image(
            f"results/output{i}.jpeg", image, cmin=0, cmax=1)

        odak.learn.tools.save_image(
            f"results/groundtruth{i}.jpeg", label, cmin=0, cmax=1)


if __name__ == "__main__":
    network = UNet()
    network.load_state_dict(torch.load("simple_unet.pth"))
    network.to(DEVICE)
    testing_loader = DataLoader(DiffuserCam(
        "/cs/student/projects1/2020/sonanuga/dataset").test_dataset)
    validate(testing_loader, network)
