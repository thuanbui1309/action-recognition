import os
import shutil

os.makedirs("data/HGP_phone_hand/", exist_ok=True)
os.makedirs("data/HGP_phone_hand/images", exist_ok=True)
labels_new = "data/HGP_phone_hand/labels"
os.makedirs(labels_new, exist_ok=True)

labels_old = os.listdir("data/HGP/labels_old")

for split in labels_old:
    split_labels = os.listdir(f"data/HGP/labels_old/{split}")
    os.makedirs(f"data/HGP_phone_hand/images/{split}", exist_ok=True)
    os.makedirs(f"{labels_new}/{split}", exist_ok=True)

    for annot in split_labels:
        data = []
        with open(f"data/HGP/labels_old/{split}/{annot}", "r") as f:
            for line in f.readlines():
                row = line.split(" ")

                if row[0] == "0":
                    data.append(row)
                elif row[0] == "2":
                    row[0] = "1"
                    data.append(row)

                if len(data) > 0:
                    with open(f"{labels_new}/{split}/{annot}", "w") as f:
                        for row in data:
                            f.write(" ".join(row))

                    shutil.copy(f"data/HGP/images/{split}/{annot[:-4]}.png", f"data/HGP_phone_hand/images/{split}/{annot[:-4]}.png")