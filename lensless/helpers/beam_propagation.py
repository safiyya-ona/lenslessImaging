import sys
import torch
import odak
import numpy as np
import argparse

LOWER_WAVELENGTH = 400e-9
UPPER_WAVELENGTH = 750e-9
DEFAULT_WAVELENGTHS = np.linspace(LOWER_WAVELENGTH, UPPER_WAVELENGTH, 300)
PI_MULT = 100 * odak.pi


class Simulation:
    def __init__(
        self,
        wavelengths=DEFAULT_WAVELENGTHS,
        pixel_pitch=8e-6,
        resolution=[270, 480],
        propagation_type="Bandlimited Angular Spectrum",
    ) -> None:
        self.wavelengths = wavelengths
        self.pixel_pitch = pixel_pitch
        self.resolution = resolution
        self.propagation_type = propagation_type
        self.anchor_wavelength = 532e-9
        self.diffuser_phase = torch.rand(*resolution) * PI_MULT
        self.diffuser_mask = None

    def simulate(
        self, input_field: torch.Tensor, distance_to_camera=0.1, distance_to_sensor=4e-3
    ) -> torch.Tensor:
        total_image_sensor = torch.zeros_like(input_field, dtype=torch.float32).to(
            input_field.device
        )
        for wavelength in self.wavelengths:
            self.diffuser_mask = odak.learn.wave.generate_complex_field(
                torch.ones_like(self.diffuser_phase).to(input_field.device),
                self.diffuser_phase.to(input_field.device)
                * (wavelength / self.anchor_wavelength),
            ).to(input_field.device)
            wavenumber = odak.wave.wavenumber(wavelength)
            field_before_diffuser = odak.learn.wave.propagate_beam(
                input_field,
                wavenumber,
                distance_to_camera,
                self.pixel_pitch,
                wavelength,
                self.propagation_type,
                [True, False, True],
            ).to(input_field.device)

            field_after_diffuser = (field_before_diffuser * self.diffuser_mask).to(
                input_field.device
            )
            field_image_sensor = odak.learn.wave.propagate_beam(
                field_after_diffuser,
                wavenumber,
                distance_to_sensor,
                self.pixel_pitch,
                wavelength,
                self.propagation_type,
                [True, False, True],
            ).to(input_field.device)

            image_sensor = (
                odak.learn.wave.calculate_amplitude(field_image_sensor) ** 2
            ).to(input_field.device)
            total_image_sensor += image_sensor / len(self.wavelengths)
        return total_image_sensor

    def diffuse_rgb_image(
        self, image: torch.Tensor, distance_to_camera=0.1, distance_to_sensor=4e-3
    ) -> torch.Tensor:
        # Diffuse an RGB image from image tensor
        diffused_rgb_image = torch.zeros_like(image, dtype=torch.float32)
        for i in range(3):
            if torch.cuda.is_available():
                current_channel = image[i].cuda()

            input_field = get_channel_input_field(current_channel).to(
                current_channel.device
            )
            diffused_rgb_image[i] = self.simulate(
                input_field, distance_to_camera, distance_to_sensor
            )
        return diffused_rgb_image


def get_image_input_field(image: torch.Tensor):
    # Generate input field for green channel of image
    input_amplitude = image[1]
    input_phase = torch.rand_like(input_amplitude)
    input_field = odak.learn.wave.generate_complex_field(input_amplitude, input_phase)
    return input_field


def get_channel_input_field(image: torch.Tensor):
    # Generate input field for given channel
    input_amplitude = image
    input_phase = torch.rand_like(input_amplitude)
    input_field = odak.learn.wave.generate_complex_field(input_amplitude, input_phase)
    return input_field


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Input filename to diffuse")
    parser.add_argument("--image_path", type=str, help="Path to image")
    parser.add_argument("--filename", type=str, help="Filename to save to")

    return parser.parse_args()


def main():
    args = parse_arguments()
    image = odak.learn.tools.load_image(
        args.image_path, normalizeby=255.0, torch_style=True
    )

    sim = Simulation(resolution=image.shape[1:])
    rgb_diffused = sim.diffuse_rgb_image(image)
    odak.learn.tools.save_image(args.filename, rgb_diffused, cmin=0, cmax=1)


if __name__ == "__main__":
    sys.exit(main())
