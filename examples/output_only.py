from utils import get_files, get_files_rigid_bodies
import shutil
import sys
import os
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import h5py

output_dir = sys.argv[1]
files = get_files_rigid_bodies(output_dir)
print(files)

# TODO add one more condition to skip this plot and go to direct comparision plots
if len(files) > 0:
    # print(directory_name+files[0])
    fn_simu = []
    time_simu = []
    for f_name in files[0:10]:
        f = h5py.File(os.path.join(output_dir, f_name), "r")
        # print("torque", f["torque_cm"][0])
        # print("ang mom", f["ang_mom_cm"][0])
        # rot_mat = f["rot_mat_cm"][0]
        print("ang vel", f["w_cm"][0])
        moi = np.asarray(f["moi_inv_global_mat_cm"][0])
        moi.resize(3, 3)
        print(moi)
        # print("MOI inv", f["moi_inv_global_mat_cm"][0])
        # print("MOI inv", f["moi_inv_global_mat_cm"][0])

        # print("ang vel", f["w_cm"][0])
        # fn_simu.append(f["torque_cm"][1][0] / 1e3)
        # time_simu.append(f.attrs["Time"] / 1e-6)

    # # save the simulated data in a npz file in the case folder
    # res_npz = os.path.join(self.input_path(name, "results.npz"))
    # np.savez(res_npz,
    #             time_simu=time_simu,
    #             fn_simu=fn_simu)

    # plt.scatter(time_analy, fn_analy, label="Analytical")
    # plt.plot(time_simu, fn_simu, "^-", label="Cabana DEM solver")
    # plt.legend()
    # plt.savefig(self.input_path(name, "fn_vs_time.pdf"))
