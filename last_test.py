import os
import cv2
import ast
import numpy as np
import nibabel as nib
from utils import pred2nii,filter

except_patient=["0002629740"]
patient_dirs="/mnt/hdd1/zhulu/blood_stage2/blood_anno/PNG"
patient_dir="Patient-0018081436"

pred2nii(patient_dirs,patient_dir)

pred_txt="secondStage/all_samples_predict_Patient.txt"
filter(pred_txt)
    