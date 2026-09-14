import pandas as pd

df = pd.read_excel("/rds/general/user/ka325/home/synthseg/SynthSeg/Cleaned_output.xlsx")

def create_label(x):

    x = str(x)

    if "Cognitively normal" in x:
        return 0
    else:
        return 1

    return None

df["Label"] = df["dx1"].apply(create_label)

df = df[df["Label"].notna()]

print(df["Label"].value_counts())

df.to_excel(
    "ML_dataset.xlsx",
    index=False
)

print("Saved ML_dataset.xlsx")
