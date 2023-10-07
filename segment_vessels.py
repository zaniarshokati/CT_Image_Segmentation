import csv
import glob

from utils import *

INPUT_PATH = './Images/slice*.nii.gz'
OUTPUT_PATH = './Vessels/'
FIGURES_PATH = './Figures/'
OVERLAY_PATH = './Vessel_overlayed/'
paths = sorted(glob.glob(INPUT_PATH))
my_file = open('vessel_volumes.csv', 'w')
lung_areas_csv = []
ratios = []

create_directory(OUTPUT_PATH)
create_directory(OVERLAY_PATH)
create_directory(FIGURES_PATH)

for c, exam_path in enumerate(paths):
    img_name = exam_path.split("/")[-1].split('.nii')[0]
    vessel_name = OUTPUT_PATH + img_name + "_vessel_only_mask"
    overlay_name = OVERLAY_PATH + img_name + "_vessels"

    ct_img = nib.load(exam_path)
    pixdim = extract_pixel_dimensions(ct_img)
    ct_numpy = ct_img.get_fdata()

    contours = segment_intensity(ct_numpy, -1000, -300)

    lungs_contour = find_lung_contours(contours)
    lung_mask = create_mask_from_polygon(ct_numpy, lungs_contour)

    lung_area = compute_lung_area(lung_mask, extract_pixel_dimensions(ct_img))

    vessels_only = create_vessel_mask(lung_mask, lungs_contour, ct_numpy, denoise=True)

    overlay_image_with_mask(ct_numpy, vessels_only)
    plt.title('Overlayed plot')
    plt.savefig(overlay_name)
    plt.close()

    save_nifty_binary_mask(vessels_only, vessel_name, affine_matrix=ct_img.affine)

    vessel_area = compute_lung_area(vessels_only, extract_pixel_dimensions(ct_img))
    ratio = (vessel_area / lung_area) * 100
    print(img_name, 'Vessel %:', ratio)
    lung_areas_csv.append([img_name, lung_area, vessel_area, ratio])
    ratios.append(ratio)

# Save data to csv file
with my_file:
    writer = csv.writer(my_file)
    writer.writerows(lung_areas_csv)
