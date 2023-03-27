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
        diffused, propagated, lensed = data
        diffused = diffused.to(device)
        propagated = propagated.to(device)
        lensed = lensed.to(device)
        output = model(propagated)

        image = torch.permute(torch.squeeze(output), (2, 1, 0))
        label = torch.permute(torch.squeeze(lensed), (2, 1, 0))

        odak.learn.tools.save_image(
            f"results_rgb/{i}output.jpeg", image, cmin=0, cmax=1)

        odak.learn.tools.save_image(
            f"results_rgb/{i}groundtruth.jpeg", label, cmin=0, cmax=1)


if __name__ == "__main__":
    network = UNet(in_channels=3, out_channels=3)  # for rgb channels
    network.load_state_dict(torch.load("beam_prop_unet_rgb_2.pth"))
    network.to(DEVICE)
    testing_loader = DataLoader(DiffuserCam(
        DIFFUSERCAM_DIR, training=False, testing=True).test_dataset)
    validate(testing_loader, network)
