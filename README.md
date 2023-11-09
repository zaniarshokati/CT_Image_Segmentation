
# Lung and Vessel Segmentation and Volume Analysis

This repository contains Python code for segmenting and analyzing lung and vessel volumes in medical images. The code is designed to work with NIfTI (Neuroimaging Informatics Technology Initiative) images and includes the following functionalities:

- **Lung Segmentation**: Segments lung regions in CT images based on intensity values, creating binary lung masks.

- **Vessel Segmentation**: Segments vessel regions in CT images based on intensity values, creating binary vessel masks. Includes an optional denoising step to remove vessels close to lung contours.

- **Volume Analysis**: Computes the area of lung regions in square millimeters (mm²) and provides the option to save the results in a CSV file.

## How to Use

To use this code, follow these steps:

1. Clone the repository or download the source code from [this link](https://github.com/zaniarshokati/CT_Image_Segmentation/).

2. Make sure you have the necessary dependencies installed. You can find a list of required libraries in the `requirements.txt` file.

3. Organize your NIfTI image data as follows:
   - Place your CT image slices in a directory (e.g., `./Images/`) with filenames like `slice*.nii.gz`.
   
4. Update the configuration in the respective Python files if needed:
   - In the `segment_lungs.py` and `segment_vessels.py` files, set the `INPUT_PATH`, `OUTPUT_PATH`, `CONTOUR_PATH`, and `OUTPUT_CSV_PATH` to match your desired file locations and names.
   
5. Run the code using the following command for lung segmentation:
   ```bash
   python segment_lungs.py

## File Descriptions

- `segment_lungs.py`: Python script for lung segmentation and volume analysis.
- `segment_vessels.py`: Python script for vessel segmentation and volume analysis.
- `utils.py`: Python module containing utility functions and classes for visualization, contour extraction, and NIfTI image processing.


## Acknowledgements

This code was developed with reference to [The AI Summer's Medical Image Analysis Tutorial](https://theaisummer.com/medical-image-python/), which provides valuable insights into medical image processing with Python.

## Dependencies

. Python 3.x
. Numpy
. nibabel
. matplotlib
. PIL
. scipy
. skimage

## License

This project is licensed under the MIT License - see the LICENSE file for details.