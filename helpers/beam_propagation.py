import sys
import torch
import odak
import numpy as np
import argparse


class Simulation:
    def __init__(self, wavelengths=[532e-9], pixel_pitch=8e-6, resolution=[1080, 1920], propagation_type='Bandlimited Angular Spectrum') -> None:
        self.wavelengths = wavelengths
        self.pixel_pitch = pixel_pitch
        self.resolution = resolution
        self.wavenumber = odak.wave.wavenumber(self.wavelengths[0])
        self.propagation_type = propagation_type
        self.diffuser_phase = torch.rand(
            resolution[0], resolution[1]) * 2 * odak.pi  # multipled by 2pi to diffuse
        self.diffuser_mask = odak.learn.wave.generate_complex_field(
            torch.ones_like(self.diffuser_phase), self.diffuser_phase)

    def simulate(self, input_field, distance_to_camera=0.1, distance_to_sensor=4e-3):
        total_image_sensor = torch.zeros_like(input_field)
        for wavelength in self.wavelengths:
            wavenumber = odak.wave.wavenumber(wavelength)
            field_before_diffuser = odak.learn.wave.propagate_beam(
                input_field, wavenumber, distance_to_camera, self.pixel_pitch, wavelength, self.propagation_type, [True, False, True])

            field_after_diffuser = field_before_diffuser * self.diffuser_mask
            field_image_sensor = odak.learn.wave.propagate_beam(
                field_after_diffuser, wavenumber, distance_to_sensor, self.pixel_pitch, wavelength, self.propagation_type, [True, False, True])

            image_sensor = odak.learn.wave.calculate_amplitude(
                field_image_sensor) ** 2

            total_image_sensor += image_sensor / len(self.wavelengths)
        return total_image_sensor


def diffuse_image(image_filename, num_wavelengths=1000):
    # Define an input field from an image
    input_field = get_image_input_field(image_filename)
    wavelengths = np.linspace(400e-9, 750e-9, num_wavelengths)
    simulator = Simulation(wavelengths=wavelengths.tolist())
    image_sensor = simulator.simulate(
        input_field, distance_to_camera=0.1, distance_to_sensor=4e-3)
    return image_sensor


def get_image_input_field(image_filename):
    # Load image.
    image = odak.learn.tools.load_image(
        image_filename, normalizeby=255., torch_style=True)
    # Define an input field.
    input_amplitude = image[1]
    input_phase = torch.rand_like(input_amplitude)
    input_field = odak.learn.wave.generate_complex_field(
        input_amplitude, input_phase)
    return input_field


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Input filename to diffuse")
    parser.add_argument("--image_path", type=str, help="Path to image")

    return parser.parse_args()


def main():
    args = parse_arguments()
    image_sensor = diffuse_image(args.image_path)
    odak.learn.tools.save_image(
        'diffused_image.png', image_sensor, cmin=0, cmax=1)


if __name__ == '__main__':
    sys.exit(main())
