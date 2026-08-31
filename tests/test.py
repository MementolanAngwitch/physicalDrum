from physicalDrum.paths import OUTPUT, ensure_dirs
import physicalDrum.drum as drum 
#import physicalDrum.experimental as drum
import matplotlib.pyplot as plt 
from datetime import datetime
import numpy as np

ensure_dirs()
run = OUTPUT / datetime.now().strftime("%Y%m%d-%H%M%S")
run.mkdir(parents=True)

table, props= drum.modes()
#strike = drum.strike(mode)
#sq_strike = drum.sequential_strike(mode)
sr =48000
#drum.write('strike.wav', strike, sr)
#drum.write('thu_glide.wav', seq_strike[0],sr)

# greaterThanOne = True
# for i in range(len(seq_strike[3])):
# 	if seq_strike[3][i] < 1:
# 		greaterThanOne = False
# if greaterThanOne:
# 	print("gamma >= 1 for every sample")

# plt.plot(seq_strike[3])
# plt.xlim(0,1440)
# plt.savefig("gamma")
# plt.show()

# y_s, *_ = drum.sequential_strike(mode, P=4.25e-3, normalize=False)
# y_h, *_ = drum.sequential_strike(mode, P=3.46e-2, normalize=False)
# peak = np.abs(y_h).max()
# drum.write('soft_strike.wav', 0.95*y_s/peak, sr)
# drum.write('hard_strike.wav', 0.95*y_h/peak, sr)

# print("soft_strike and hard_strike wav files generated")

# y_s, *_ = drum.sequential_strike(table, props, P=4.25e-3, normalize=False)
# y_h, *_ = drum.sequential_strike(table, props, P=3.46e-2, normalize=False)
# peak = np.abs(y_h).max()
# # drum.write('modes_c_soft.wav', 0.95*y_s/peak, sr)
# drum.write('modes_c_hard.wav', 0.95*y_h/peak, sr)

# print("soft_strike and hard_strike wav files generated")
# drum.write(run / "soft.wav", 0.95 * y_s / peak, sr)
# drum.write(run / "hard.wav", 0.95 * y_h / peak, sr)

# y_s, *_ = drum.sequential_strike(table, props, P=4.25e-3, normalize=False)
# peak = np.abs(y_s).max()
# drum.write('new_modes()_soft.wav', 0.95*y_s/peak, sr)

for L in [1, 4, 8, 16, 32]:
    y, S, gc, ga = drum.sequential_strike(table, props, P=3.46e-2, L_slow=L)
    print(L, ga.max(), S.max())