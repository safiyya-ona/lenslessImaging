import torch
from torch.utils.data import DataLoader
import odak
from lensless.models.unets.simple_unet import UNet as SimpleUNet
from lensless.models.unets.attention_unet import AttentionUNet
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
        output = model(diffused)

        image = torch.permute(torch.squeeze(output), (2, 1, 0))
        label = torch.permute(torch.squeeze(lensed), (2, 1, 0))

        odak.learn.tools.save_image(
            f"results_results/{i}output.jpeg", image, cmin=0, cmax=1)

        odak.learn.tools.save_image(
            f"results_results/{i}groundtruth.jpeg", label, cmin=0, cmax=1)


if __name__ == "__main__":
    network = SimpleUNet(in_channels=3, out_channels=3)
    network.load_state_dict(torch.load("simple_model_diffuser.pth"))
    network.to(DEVICE)
    testing_loader = DataLoader(DiffuserCam(
        DIFFUSERCAM_DIR, training=False, testing=True).test_dataset)
    validate(testing_loader, network)
