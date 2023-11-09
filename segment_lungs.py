import glob
import csv
import os
import nibabel as nib
from utils import *

class LungVolumeAnalyzer:
    def __init__(self, input_path, output_path, contour_path, output_csv_path):
        self.visualizer = Visualization()
        self.lung = Lung()
        self.nifty = Nifty()

        self.input_path = input_path
        self.output_path = output_path
        self.contour_path = contour_path
        self.output_csv_path = output_csv_path
        
    def process_image(self, path):
        img_name = os.path.basename(path).split(".nii")[0]
        out_mask_name = os.path.join(self.output_path, f"{img_name}_mask")
        contour_name = os.path.join(self.contour_path, f"{img_name}_contour")

        ct_img = nib.load(path)
        ct_numpy = ct_img.get_fdata()

        contours = self.lung.segment_intensity(ct_numpy, lower_bound=-1000, upper_bound=-300)
        lungs = self.lung.find_lung_contours(contours)

        self.visualizer.display_contours(ct_numpy, lungs, title=contour_name, save=True)
        lung_mask = self.lung.create_mask_from_polygon(ct_numpy, lungs)
        self.nifty.save_nifty_binary_mask(lung_mask, out_mask_name, ct_img.affine)

        lung_area = self.lung.compute_lung_area(lung_mask, ct_img)
        return img_name, lung_area

    def analyze_images(self):
        paths = sorted(glob.glob(self.input_path))
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.contour_path, exist_ok=True)

        lung_areas = []

        for path in paths:
            img_name, lung_area = self.process_image(path)
            lung_areas.append([img_name, lung_area])

        with open(self.output_csv_path, "w", newline="") as my_file:
            writer = csv.writer(my_file)
            writer.writerows(lung_areas)

if __name__ == "__main__":
    INPUT_PATH = "./Images/slice*.nii.gz"
    OUTPUT_PATH = "./Lungs/"
    CONTOUR_PATH = "./Contours/"
    OUTPUT_CSV_PATH = "lung_volumes.csv"
    

    analyzer = LungVolumeAnalyzer(INPUT_PATH, OUTPUT_PATH, CONTOUR_PATH, OUTPUT_CSV_PATH)
    analyzer.analyze_images()
