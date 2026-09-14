import os
import pandas as pd
import subprocess
from multiprocessing import Pool

EXCEL_FILE = "/rds/general/user/ka325/home/synthseg/SynthSeg/Clinical_diagnosis.xlsx"

MRI_FOLDER = "/rds/general/user/ka325/home/synthseg/SynthSeg/OASIS_MRI"

EC_SCRIPT = "/rds/general/user/ka325/home/synthseg/SynthSeg/EC_Final.py"

OUTPUT_FILE = "/rds/general/user/ka325/home/synthseg/SynthSeg/Clinical_diagnosis_completed.xlsx"


def process_subject(args):

    index, row = args

    mr_id = str(row["MRI_session_label"])

    try:

        parts = mr_id.split("_")

        subject_id = parts[0]
        session_id = parts[2]

        matching_file = None

        for file in os.listdir(MRI_FOLDER):

            if (
                subject_id in file
                and session_id in file
                and file.endswith(".nii.gz")
            ):
                matching_file = os.path.join(MRI_FOLDER, file)
                break

        if matching_file is None:
            print(f"❌ MRI not found: {mr_id}")
            return None

        print(f"✅ Processing: {mr_id}")

        cmd = (
            "module load Python/3.8.6-GCCcore-10.2.0 && "
            f"python {EC_SCRIPT} {matching_file}"
        )

        result = subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"❌ EC extraction failed: {mr_id}")
            return None

        ec_volume = None
        brain_volume = None
        ec_percentage = None

        for line in result.stdout.splitlines():

            if line.startswith("EC_VOLUME="):
                ec_volume = float(line.split("=")[1])

            elif line.startswith("BRAIN_VOLUME="):
                brain_volume = float(line.split("=")[1])

            elif line.startswith("EC_PERCENTAGE="):
                ec_percentage = float(line.split("=")[1])

        return (
            index,
            brain_volume,
            ec_volume,
            ec_percentage
        )

    except Exception as e:

        print(f"❌ Error in {mr_id}")
        print(e)

        return None

# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":

    df = pd.read_excel(EXCEL_FILE)

    print(f"Loaded {len(df)} subjects")

    tasks = [(index, row) for index, row in df.iterrows()]

    completed = 0

    with Pool(processes=4) as pool:

        for result in pool.imap_unordered(process_subject, tasks):

            if result is None:
                continue

            index, brain_volume, ec_volume, ec_percentage = result

            df.at[index, "Brain_volume"] = brain_volume
            df.at[index, "EC_volume"] = ec_volume
            df.at[index, "EC_percentage"] = ec_percentage

            completed += 1

            # Save every 10 completed subjects
            if completed % 10 == 0:

                df.to_excel(OUTPUT_FILE, index=False)

                print(
                    f"💾 Progress saved | "
                    f"{completed}/{len(df)} subjects completed"
                )

    # Final save
    df.to_excel(OUTPUT_FILE, index=False)

    print("\n✅ Finished!")
    print(f"✅ Saved dataset: {OUTPUT_FILE}")