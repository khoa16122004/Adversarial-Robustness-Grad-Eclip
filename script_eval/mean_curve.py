import os
import json
import numpy as np
import matplotlib.pyplot as plt

result_dir = "/home/jovyan/clean_causual/eclip"
mode = "deletion"

aucs = []
all_curves = []

for folder_name in sorted(os.listdir(result_dir)):
    folder_path = os.path.join(result_dir, folder_name)

    if not os.path.isdir(folder_path):
        continue

    data_path = os.path.join(folder_path, f"{mode}_information.json")
    if not os.path.exists(data_path):
        continue

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    aucs.append(data[f"{mode}_auc"])
    all_curves.append(np.asarray(data[f"{mode}_curve"], dtype=np.float32))

all_curves = np.stack(all_curves)

# statistics
mean_auc = np.mean(aucs)
std_auc = np.std(aucs)

mean_curve = all_curves.mean(axis=0)
std_curve = all_curves.std(axis=0)

print(f"Number of samples : {len(aucs)}")
print(f"Mean AUC          : {mean_auc:.4f}")
print(f"Std AUC           : {std_auc:.4f}")

# plot
x = np.linspace(0, 1, len(mean_curve))

plt.figure(figsize=(6,4))
plt.plot(x, mean_curve, label="Mean Curve", linewidth=2)
plt.fill_between(
    x,
    mean_curve - std_curve,
    mean_curve + std_curve,
    alpha=0.25,
    label="±1 std"
)

plt.xlabel("Fraction of pixels removed")
plt.ylabel("Confidence")
plt.title(f"Average {mode.capitalize()} Curve")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()