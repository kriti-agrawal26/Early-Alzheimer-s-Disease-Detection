import sys
import os
import numpy as np
import nibabel as nib
import subprocess

# -------- Step 1: Input --------
if len(sys.argv) < 2:
    print("Usage: python run_ec.py <input_mri>")
    sys.exit(1)

input_mri = sys.argv[1]
print("Input MRI:", input_mri)

# -------- Step 2: Run SynthSeg (parcellation) --------
base_name=os.path.basename(input_mri).replace('.nii.gz','')

output_parc = base_name + "_parc.nii.gz"

cmd = [
    "python",
    "scripts/commands/SynthSeg_predict.py",
    "--i", input_mri,
    "--o", output_parc,
    "--parc",
    "--cpu",
    "--threads", "2",
    "--fast",
    "--crop", "160", "160", "160"
]

subprocess.run(cmd, check=True)
#print("✅ Parcellation complete")

# -------- Step 3: Load data --------
t1 = nib.load(input_mri)
parc = nib.load(output_parc)

t1_data = t1.get_fdata()
parc_data = parc.get_fdata()

#print("Labels:", np.unique(parc_data))

# -------- Step 4: Extract EC --------
ec_mask = (parc_data == 1006) | (parc_data == 2006)
#print("EC voxels:", ec_mask.sum())

# -------- Step 5: Overlay --------
combined = t1_data.copy()
combined[ec_mask] = combined.max()

combined_img = nib.Nifti1Image(combined, t1.affine, t1.header)
nib.save(combined_img, os.path.join("Processed_images",base_name+"_EChigh.nii.gz"))

# -------- Step 6: Volume calculation --------
voxel_dims = parc.header.get_zooms()
voxel_volume = voxel_dims[0] * voxel_dims[1] * voxel_dims[2]

EC_volume = ec_mask.sum() * voxel_volume
brain_volume = np.sum(parc_data > 0) * voxel_volume
EC_percentage = (EC_volume / brain_volume) * 100

print(f"EC_VOLUME={EC_volume}")
print(f"BRAIN_VOLUME={brain_volume}")
print(f"EC_PERCENTAGE={EC_percentage}")

if os.path.exists(output_parc):
    os.remove(output_parc)
    #print(f"{output_parc} deleted")