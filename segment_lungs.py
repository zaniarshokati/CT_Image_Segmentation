import glob
import csv
from utils import *

INPUT_PATH = './Images/slice*.nii.gz'
OUTPUT_PATH = './LUNGS/'
CONTOUR_PATH = './Contours/'
paths = sorted(glob.glob(INPUT_PATH))
my_file = open('lung_volumes.csv', 'w')
lung_areas = []
make_dirs(OUTPUT_PATH)
make_dirs(CONTOUR_PATH)

for c, exam_path in enumerate(paths):
    img_name = exam_path.split("/")[-1].split('.nii')[0]
    out_mask_name = OUTPUT_PATH + img_name + "_mask"
    contour_name = CONTOUR_PATH + img_name + "_contour"

    ct_img = nib.load(exam_path)
    pixdim = extract_pixel_dimensions(ct_img)
    ct_numpy = ct_img.get_fdata()

    contours = intensity_seg(ct_numpy, min=-1000, max=-300)

    lungs = find_lungs(contours)
    show_contour(ct_numpy, lungs, contour_name,save=True)
    lung_mask = create_mask_from_polygon(ct_numpy, lungs)
    save_nifty(lung_mask, out_mask_name, ct_img.affine)

    lung_area = compute_area(lung_mask, extract_pixel_dimensions(ct_img))
    lung_areas.append([img_name,lung_area]) # int is ok since the units are already mm^2


with my_file:
    writer = csv.writer(my_file)
    writer.writerows(lung_areas)
