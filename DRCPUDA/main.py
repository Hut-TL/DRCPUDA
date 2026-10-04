import os
import torch
from datasets import dataset
from training.train import pre_training, fine_tuning, test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path_hust = "datasets/HUST_Bearing_raw_data/"
path_pu = "datasets/PU_data_txt/"

unlabel = 5
num_classes = 5
batch_size = 128

s1_label_list = [0, 1, 2, 3, 4]
s2_label_list = [0, 1, 2, 3, 4]
s3_label_list = [0, 1, 2, 3, 4, 5]
s4_label_list = [0, 1, 2, 3, 4]

hust_filenames1 = ["H_65Hz.xls", "0.5X_I_65Hz.xls", "I_65Hz.xls", "0.5X_B_65Hz.xls", "B_65Hz.xls",
                   "0.5X_O_65Hz.xls", "O_65Hz.xls", "0.5X_C_65Hz.xls", "C_65Hz.xls"]

hust_filenames2 = ["H_70Hz.xls", "0.5X_I_70Hz.xls", "I_70Hz.xls", "0.5X_B_70Hz.xls", "B_70Hz.xls",
                   "0.5X_O_70Hz.xls", "O_70Hz.xls", "0.5X_C_70Hz.xls", "C_70Hz.xls"]

hust_filenames3 = ["H_75Hz.xls", "0.5X_I_75Hz.xls", "I_75Hz.xls", "0.5X_B_75Hz.xls", "B_75Hz.xls",
                   "0.5X_O_75Hz.xls", "O_75Hz.xls", "0.5X_C_75Hz.xls", "C_75Hz.xls"]

hust_filenames4 = ["H_80Hz.xls", "0.5X_I_80Hz.xls", "I_80Hz.xls", "0.5X_B_80Hz.xls", "B_80Hz.xls",
                   "0.5X_O_80Hz.xls", "O_80Hz.xls", "0.5X_C_80Hz.xls", "C_80Hz.xls"]

pu_filenames1 = ["0_N09_M07_F10_K004_1.txt", "1_N09_M07_F10_KI04_1.txt", "2_N09_M07_F10_KI18_1.txt",
                 "3_N09_M07_F10_KI16_1.txt",
                 "4_N09_M07_F10_KA15_1.txt", "5_N09_M07_F10_KA16_1.txt",
                 "6_N09_M07_F10_KB27_1.txt", "7_N09_M07_F10_KB23_1.txt", "8_N09_M07_F10_KB24_1.txt"]
pu_filenames2 = ["0_N15_M01_F10_K004_1.txt", "1_N15_M01_F10_KI04_1.txt", "2_N15_M01_F10_KI18_1.txt",
                 "3_N15_M01_F10_KI16_1.txt",
                 "4_N15_M01_F10_KA15_1.txt", "5_N15_M01_F10_KA16_1.txt",
                 "6_N15_M01_F10_KB27_1.txt", "7_N15_M01_F10_KB23_1.txt", "8_N15_M01_F10_KB24_1.txt"]
pu_filenames3 = ["0_N15_M07_F04_K004_1.txt", "1_N15_M07_F04_KI04_1.txt", "2_N15_M07_F04_KI18_1.txt",
                 "3_N15_M07_F04_KI16_1.txt",
                 "4_N15_M07_F04_KA15_1.txt", "5_N15_M07_F04_KA16_1.txt",
                 "6_N15_M07_F04_KB27_1.txt", "7_N15_M07_F04_KB23_1.txt", "8_N15_M07_F04_KB24_1.txt"]
pu_filenames4 = ["0_N15_M07_F10_K004_1.txt", "1_N15_M07_F10_KI04_1.txt", "2_N15_M07_F10_KI18_1.txt",
                 "3_N15_M07_F10_KI16_1.txt",
                 "4_N15_M07_F10_KA15_1.txt", "5_N15_M07_F10_KA16_1.txt",
                 "6_N15_M07_F10_KB27_1.txt", "7_N15_M07_F10_KB23_1.txt", "8_N15_M07_F10_KB24_1.txt"]


def main():
    print(f"Using device: {device}")
    s1_train, s1_test = dataset.get_dataloader(
        path_pu,
        [pu_filenames2[i] for i in s2_label_list],
        [0.8, 0.2],
        batch_size=batch_size,
        shuffle=True,
        separator="\n",
        fixed_label=s2_label_list,
        unlabel=unlabel
    )
    s2_train, s2_test = dataset.get_dataloader(
        path_pu,
        [pu_filenames3[i] for i in s3_label_list],
        [0.8, 0.2],
        batch_size=batch_size,
        shuffle=True,
        separator="\n",
        fixed_label=s3_label_list,
        unlabel=unlabel
    )

    hcos, hd = pre_training(s1_train, s1_test, num_classes, epochs=1)
    fine_tuning(s2_train, num_classes, hcos, epochs=2)
    test(s2_test, hd, num_classes)


if __name__ == "__main__":
    main()
