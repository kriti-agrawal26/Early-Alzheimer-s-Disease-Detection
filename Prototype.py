import os
import subprocess
import joblib
import pandas as pd

# -----------------------------------------------------------
# INPUT
# -----------------------------------------------------------

print("===== Alzheimer's Assessment Prototype =====")

name = input("Enter Patient Name: ").strip()

if not name:
    print("Error: Patient name cannot be empty")
    raise SystemExit

mri_file_FH = input("Enter MRI File Name (without .nii.gz): ").strip()

mri_file = mri_file_FH + ".nii.gz"

age = int(input("Enter Age: "))

if age < 40 or age > 120:
    print("Error: Please enter a valid age between 40 and 120")
    raise SystemExit

mmse = int(input("Enter MMSE Score: "))

if mmse < 0 or mmse > 30:
    print("Error: MMSE score must be between 0 and 30")
    raise SystemExit

# -----------------------------------------------------------
# MRI LOOKUP
# -----------------------------------------------------------

MRI_FOLDER = "/rds/general/user/ka325/home/synthseg/SynthSeg/OASIS_MRI"

input_mri = os.path.join(MRI_FOLDER, mri_file)

if not os.path.exists(input_mri):
    print("Error: MRI file not found")
    raise SystemExit

# -----------------------------------------------------------
# RUN EC EXTRACTION
# -----------------------------------------------------------

cmd = (
    "module load Python/3.8.6-GCCcore-10.2.0 && "
    "python /rds/general/user/ka325/home/synthseg/SynthSeg/EC_Final.py "
    + input_mri
)

result_vol_per = subprocess.run(
    cmd,
    shell=True,
    executable="/bin/bash",
    capture_output=True,
    text=True
)

if result_vol_per.returncode != 0:
    print("Error: EC extraction failed")
    print(result_vol_per.stderr)
    raise SystemExit

# -----------------------------------------------------------
# EXTRACT EC VALUES
# -----------------------------------------------------------

ec_volume = None
brain_volume = None
ec_percentage = None

for line in result_vol_per.stdout.splitlines():

    if line.startswith("EC_VOLUME="):
        ec_volume = float(line.split("=")[1])

    elif line.startswith("BRAIN_VOLUME="):
        brain_volume = float(line.split("=")[1])

    elif line.startswith("EC_PERCENTAGE="):
        ec_percentage = float(line.split("=")[1])

if ec_volume is None:
    print("Error: Could not extract MRI biomarkers")
    raise SystemExit

# -----------------------------------------------------------
# LOAD ML MODEL
# -----------------------------------------------------------

model = joblib.load(
    "/rds/general/user/ka325/home/synthseg/SynthSeg/ad_classifier.joblib"
)

# -----------------------------------------------------------
# PREPARE FEATURES
# -----------------------------------------------------------

patient_data = pd.DataFrame(
    [[
        age,
        mmse,
        ec_volume,
        ec_percentage
    ]],
    columns=[
        "age at visit",
        "MMSE",
        "EC_volume",
        "EC_percentage"
    ]
)

# -----------------------------------------------------------
# PREDICT
# -----------------------------------------------------------

prediction = model.predict(patient_data)[0]

probabilities = model.predict_proba(patient_data)[0]

if prediction == 1:
    diagnosis = "Alzheimer's Disease"
else:
    diagnosis = "Cognitively Normal"

# -----------------------------------------------------------
# REPORT
# -----------------------------------------------------------

print("\n===================================")
print("      FINAL PATIENT REPORT")
print("===================================")

print(f"Patient Name : {name}")
print(f"Age          : {age}")
print(f"MRI File     : {mri_file}")

print("\nClinical Data")
print(f"MMSE Score   : {mmse}")

print("\nMRI Biomarkers")
print(f"EC Volume    : {ec_volume:.2f} mm³")
print(f"Brain Volume : {brain_volume:.2f} mm³")
print(f"EC Percentage: {ec_percentage:.4f} %")

print("\nMachine Learning Prediction")
print(f"Diagnosis    : {diagnosis}")

print(f"Normal Prob. : {probabilities[0] * 100:.2f}%")
print(f"AD Prob.     : {probabilities[1] * 100:.2f}%")

print("===================================")