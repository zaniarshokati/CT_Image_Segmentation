import csv
import glob
import os
import nibabel as nib
import matplotlib.pyplot as plt
from utils import *

class VesselVolumeAnalyzer:
    def __init__(self, input_path, output_path, overlay_path, output_csv_path):
        self.input_path = input_path
        self.output_path = output_path
        self.overlay_path = overlay_path
        self.output_csv_path = output_csv_path
        self.visualizer = Visualization()
        self.lung_functions = LungHandler()

    def process_image(self, exam_path):
        img_name = os.path.basename(exam_path).split(".nii")[0]
        vessel_name = os.path.join(self.output_path, f"{img_name}_vessel_only_mask")
        overlay_name = os.path.join(self.overlay_path, f"{img_name}_vessels")

        ct_img = nib.load(exam_path)
        ct_numpy = ct_img.get_fdata()

        contours = segment_intensity(ct_numpy, -1000, -300)
        lungs_contour = find_lung_contours(contours)
        lung_mask = self.lung_functions.create_mask_from_polygon(ct_numpy, lungs_contour)

        lung_area = compute_lung_area(lung_mask, extract_pixel_dimensions(ct_img))

        vessels_only = create_vessel_mask(lung_mask, lungs_contour, ct_numpy, denoise=True)
        self.visualizer.show_image_slice(vessels_only)

        overlay_image_with_mask(ct_numpy, vessels_only)
        plt.title("Overlayed plot")
        plt.savefig(overlay_name)
        plt.close()

        save_nifty_binary_mask(vessels_only, vessel_name, affine_matrix=ct_img.affine)

        vessel_area = compute_lung_area(vessels_only, extract_pixel_dimensions(ct_img))
        ratio = (vessel_area / lung_area) * 100
        print(f"{img_name} - Vessel %: {ratio:.2f}%")

        return [img_name, lung_area, vessel_area, ratio]

    def analyze_images(self):
        paths = sorted(glob.glob(self.input_path))
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.overlay_path, exist_ok=True)

        vessel_data = []

        for exam_path in paths:
            img_data = self.process_image(exam_path)
            vessel_data.append(img_data)

        with open(self.output_csv_path, "w", newline="") as my_file:
            writer = csv.writer(my_file)
            writer.writerows(vessel_data)

if __name__ == "__main__":
    INPUT_PATH = "./Images/slice*.nii.gz"
    OUTPUT_PATH = "./Vessels/"
    OVERLAY_PATH = "./Vessel_overlayed/"
    OUTPUT_CSV_PATH = "vessel_volumes.csv"

    analyzer = VesselVolumeAnalyzer(INPUT_PATH, OUTPUT_PATH, OVERLAY_PATH, OUTPUT_CSV_PATH)
    analyzer.analyze_images()
