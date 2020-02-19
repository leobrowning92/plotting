import os
import matplotlib.pyplot as plt
from plotting import mea_analysis as mea

# checks and creates output folders
if not (os.path.isdir("plots")):
    os.mkdir("plots")


# mea001 plots
smudf, vdf = mea.open_data("example_data/MEA001_postS_5V_SMU.tdms")
mea.plot_eventanalysis(smudf, gthresh=0.0045, name="MEA001")

mea.plot_signal(
    vdf,
    smudf,
    show=False,
    save="plots/mea001fulltrace",
    tr=(0, 1e10),
    remove_channels=[],
    title=False,
)

mea.plot_signal(
    vdf,
    smudf,
    show=False,
    save="plots/mea001_comparetrace",
    tr=(43, 44),
    remove_channels=[],
    title=False,
)

fig, ax = plt.subplots(figsize=(6.3, 3.15))
mea.plot_voltages(vdf, ax, tr=(43.4, 43.6))
fig.tight_layout()
plt.savefig("plots/mea001restrace.png")
plt.savefig("plots/mea001restrace.pdf")

resd = vdf.iloc[:, 3:-1].values[43400:43600]
mea.plot_principal_components(resd, save="plots/mea001respca", ls="-")
resd = vdf.sample(frac=0.1).sort_index().iloc[:, 3:-1].values[::]
mea.plot_principal_components(resd, save="plots/mea001fullpca", alpha=0.05, ends=False)
