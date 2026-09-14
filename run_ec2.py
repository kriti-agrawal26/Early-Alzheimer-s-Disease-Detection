import sys
import os
import numpy as np
import nibabel as nib
from SynthSeg.predict import predict
import matplotlib.pyplot as plt

#-------------------------------------------
#step 1 get input MRI file
#-------------------------------------------
if len(sys.argv)<2:
    print("Usage: python run_ec.py <input_mri>")
    sys.exit(1)
input_mri=sys.argv[1]
print("Input MRI:", input_mri)

#-------------------------------------------
#step 2 define output files
#-------------------------------------------
base_name=os.path.splitext(os.path.basename(input_mri))[0]
overlay_file=base_name+"_EChighl.nii.gz"
print("Final image:", overlay_file)

#-------------------------------------------
#step 3 run syhthseg
#-------------------------------------------
temp_seg = base_name+"_temp_seg.nii.gz"
#predict(input_mri,temp_seg,"models/synthseg_2.0.h5","data/labels_classes_priors/synthseg_segmentation_labels_2.0.npy")
parc_img = base_name+"_parc_seg.nii.gz"
predict(input_mri,parc_img,"models/synthseg_parc_2.0.h5","data/labels_classes_priors/synthseg_parcellation_labels.npy")
print("SynthSeg Parcellation completed")

print("SynthSeg completed")

#-------------------------------------------
#step 4 load data
#-------------------------------------------
mri_img=nib.load(input_mri)
seg_img=nib.load(parc_img)
mri=mri_img.get_fdata()
seg=seg_img.get_fdata()
print("Labels:",np.unique(seg))
#-------------------------------------------
#step 5 extract ec
#left EC=1006, Right EC=2006
#-------------------------------------------
EC_mask=(seg == 1006)|(seg == 2006)
ECsum=EC_mask.sum()
print(ECsum)

#-------------------------------------------
#step 6 create overlay image
#-------------------------------------------
overlay=mri.copy()
#highlight EC by increasing intensity
overlay[EC_mask]=np.max(mri)*1.5
overlay_img=nib.Nifti1Image(overlay, mri_img.affine, mri_img.header)
nib.save(overlay_img, overlay_file)
print("Overlay image saved:", overlay_file)

#-------------------------------------------
#step 7 compute volumes
#-------------------------------------------
voxel_dims=seg_img.header.get_zooms()
voxel_volume=voxel_dims[0]*voxel_dims[1]*voxel_dims[2]
EC_volume_mm3=np.sum(EC_mask)*voxel_volume
#brain volume=non-background voxels
brain_mask=seg>0
brain_volume_mm3=np.sum(brain_mask)*voxel_volume
EC_percentage=(EC_volume_mm3/brain_volume_mm3)*100

#-------------------------------------------
#step 8 output results
#-------------------------------------------
print("==============================")
print("EC_VOLUME:",EC_volume_mm3)
print("BRAIN_VOLUME:",brain_volume_mm3)
print("EC_PERCENTAGE:",EC_percentage)
print("==============================")

#-------------------------------------------
#cleanup
#-------------------------------------------
#os.remove(temp_seg)
#print("Temporary file removed")
