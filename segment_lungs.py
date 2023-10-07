import glob
import csv
import os
import nibabel as nib
from utils import *

INPUT_PATH = './Images/slice*.nii.gz'
OUTPUT_PATH = './LUNGS/'
CONTOUR_PATH = './Contours/'
OUTPUT_CSV_PATH = 'lung_volumes.csv'

def process_image(exam_path):
    img_name = exam_path.split("/")[-1].split('.nii')[0]
    out_mask_name = os.path.join(OUTPUT_PATH, f'{img_name}_mask')
    contour_name = os.path.join(CONTOUR_PATH, f'{img_name}_contour')

    ct_img = nib.load(exam_path)
    ct_numpy = ct_img.get_fdata()

    contours = segment_intensity(ct_numpy, lower_bound=-1000, upper_bound=-300)
    lungs = find_lung_contours(contours)

    display_contours(ct_numpy, lungs, title=contour_name, save=True)
    lung_mask = create_mask_from_polygon(ct_numpy, lungs)
    save_nifty_binary_mask(lung_mask, out_mask_name, ct_img.affine)

    lung_area = compute_lung_area(lung_mask, extract_pixel_dimensions(ct_img))
    return img_name, lung_area

def main():
    paths = sorted(glob.glob(INPUT_PATH))
    create_directory(OUTPUT_PATH)
    create_directory(CONTOUR_PATH)

    lung_areas = []

    for exam_path in paths:
        img_name, lung_area = process_image(exam_path)
        lung_areas.append([img_name, lung_area])

    with open(OUTPUT_CSV_PATH, 'w', newline='') as my_file:
        writer = csv.writer(my_file)
        writer.writerows(lung_areas)

if __name__ == "__main__":
    main()
