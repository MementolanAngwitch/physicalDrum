from scipy import special
import numpy as np 
from scipy.io import wavfile
import matplotlib.pyplot as plt 

sr = 48000

def modes(a=0.164,f_max=5000,r_strike=0.4,r_pickup=0.6,theta=np.pi/2, T_0 = 3530.6, rho = 1390, h = 2.54e-4):
	#c is the transverse wave speed on the membrane, 100m/s on 10-mil mylar
	#theta is difference between strike and pickup point
	# h is thickness in meters, T_0 is tuned tension, rho is mass density of the head material, tuned to mylar tom
	#Returns a list of (k,f,in,out,norm), (wavenumber, freq, mode shape at strike, mode at pickup, squared length of mode shape)
	m = 0
	out = []
	c = np.sqrt(T_0/(rho*h))
	while True:
		'''
		#Returns a list of ((f,m,n,j))
		zeros = special.jn_zeros(m, 64)
		f = c * zeros / (2 * np.pi * a)
		g = f < f_max 
		if not g.any():
			break
		zeros = zeros[:g.sum()]
		for i,z in enumerate(zeros):
			out.append((f[i],m,i+1,zeros[i]))
		'''
		zeros = special.jn_zeros(m,64)
		f = c * zeros / (2 * np.pi *a)
		g = f < f_max
		if not g.any():
			break
		zeros = zeros[:g.sum()]
		for i,z in enumerate(zeros):
			j = zeros[i]
			k = j/a 
			modeIn = special.jv(m, j*r_strike)
			modeOut = special.jv(m, j*r_pickup) * np.cos(theta*m)
			if m == 0: eps = 2.0 
			else: eps = 1.0
			norm = eps * (np.pi * (a ** 2)/2 ) * special.jv(m+1, j) ** 2
			out.append((k,f[i],modeIn,modeOut,norm))
		m += 1
	out.sort(key=lambda r: r[1])
	props = {'A_area': np.pi *a ** 2, 'T_0': T_0, 'sigma_mu': rho*h, 'c':c, 'h':h}
	#print(c)
	#print(out[0][1])
	return out, props



def strike(modes, dur = 2.0 , sr= 48000, w = 0.00984, amp =1.0, sigma_a=2.0, sigma_b=1.2e-4):
	'''
	sigma_a flat rate, sets the longest anything can ring 1/sigma_a
	sigma_b multiplies f^2, determines damping
	raise sigma_a   whole drum decays faster, evenly     (towel on the head)
	raise sigma_b   highs die faster than lows           duller, thuddier
	lower sigma_b   highs hang around                    bright, metallic, gong-ish

	y(t) = sum_i  A_i * sin(w_i * t) * exp(-sig_i * t)      w_i = 2*pi*f_i
	norm_i  = eps_m * (pi*a^2/2) * J_(m+1)(j)^2        eps_m = 2 if m==0 else 1
	in_i    = J_m(j * r_strike)                        r_strike = 0.6   (as fraction of a)
	out_i   = J_m(j * r_pickup) * cos(m * theta)       r_pickup = 0.7, theta = 0.6
	roll_i  = exp(-0.5 * (j * w)^2)                    w = 0.06   (stick contact radius)
	A_i     = in_i * out_i * roll_i / norm_i

	sig_i   = 2.0 + 1.2e-4 * f_i^2                     clamp to <= 0.95 * w_i
	'''
	t = np.arange(int(dur * sr)) / sr # time in seconds
	y = np.zeros_like(t) # buffer 
	'''
	for f,m,n,j in modes:
		# compute A, w, sig for this mode
		if m == 0:
			eps_m = 2.0
		else:
			eps_m = 1.0
		norm_i = eps_m * (np.pi * a**2 / 2) * special.jv(m+1, j) ** 2
		in_i = special.jv(m, j*r_strike)
		out_i = special.jv(m, j*r_pickup) * np.cos(m * theta)
		roll_i = np.exp(-0.5 * (j * sc)**2) # sc = 0.06, stick contact radius
		A_i = in_i * out_i * roll_i / norm_i
		sig_i = sigma_a + sigma_b * f ** 2# clamp to <= 0.95 * w_i
		w_i = 2 * np.pi * f 

		y += A_i * np.sin(w_i * t) * np.exp(-sig_i * t)
	'''
	for k,f,modeIn,modeOut,norm in modes:
		'''
		The head's motion is a sum of independent oscillators
		u(r,theta,t) = sum_i q_i(t) + Phi_i(r,theta)
		the shapes Phi_i are frozen, time dependency is in the coefficients q_i(t)
		q_i'' 2*sig_i*q_i + omega_i**2 * q_i = 0
		each iteration has three tasks:
		1. modeIn * roll/norm how hard the strike excited this mode, a number
		2. exp(-sig*t) * sin(omega_d*t) what this mode then does, a wave
		3. modeOut how much of it reaches the pickup
		y+=readout*excitation*waveform
		'''
		sig = sigma_a + sigma_b * f**2 # amplitude decay rate of this node, s^(-1), fitted to perception, energy decays at 2*sig
		omega = 2*np.pi*f  #undamped angular freq
		omega_d = np.sqrt(omega ** 2 - sig**2) # damped angular freq
		roll = np.exp(-0.5*(k*w)**2) #contact-width rolloff, how much of this stick's spatial spectrum lands on this mode
		A = modeIn * roll/norm * amp # excitation scalar, how of this mode the strike puts in
		env = np.exp(-sig*t) * np.sin(omega_d*t) # the mode's waveform at unit amplitude, the solution of q''+2*sqig*q'+ omega**2*q = 0
		q = env * A # the modal corrinate, the displacement of mode i over time in meters
		y += q * modeOut#readout

	return y / np.abs(y).max() # normalization

def sequential_strike(table, props, dur = 2.0 , sr= 48000, w = 0.00984, sigma_a=2.0, sigma_b=3.723e-5, 
	P=1.73e-2, E = 4.0e9, normalize = True):
	# rho is the mylar density, sigma_a/_b are damping elements, h is thickness, E is young's modulus, T_0 is the base tension
	# sigma_mu is aeral density and P is strike impulse
	n = int(dur * sr)
	y= np.zeros(n)  
	k, f, modeIn, modeOut, norm = np.array(table).T 

	sig = sigma_a + sigma_b * f**2 # amplitude decay rate of this node, s^(-1), fitted to perception, decays at 2*sig
	omega = 2*np.pi*f  #undamped angular freq
	omega_d = np.sqrt(omega ** 2 - sig**2) # damped angular freq
	roll = np.exp(-0.5*(k*w)**2) #contact-width rolloff, how much of this stick's spatial spectrum lands on this mode
	sigma_mu = props['sigma_mu']
	A = modeIn * roll/norm/omega_d * (P/sigma_mu) # excitation scalar, how of this mode the strike puts in'
	dt = 1/sr 

	# Beta is how much the tension rose Eh / (2* A_area * T_0) where E is young's modulus and h is thickness
	# T_0 = sigma * c**2
	# c = np.sqrt(T_0 / sigma_mu) check to verify c matches c in modes 
	#A_area = np.pi * a **2 
	Eh = E*props['h']

	beta = Eh / (2*props['A_area']*props['T_0'])
	kern = k**2 * norm # The S-kernal
	S_tr = np.zeros(n) # Trace
	gamma_c = np.ones(n) # computed gamma, gamma is the frequency multiplier
	gamma_a = np.ones(n) # applied gamma
	g = 1.0 # the currently applied scalar
	n_fast = int(0.001*sr) 
	L_fast = 4 # rebuild coefficients every 4 samples while i < n_fast
	L_slow = 8 # every 8 samples after that

	q_prev = np.zeros_like(A)
	q_curr = A * np.exp(-sig * dt) * np.sin(omega_d * dt)

	y[0] = 0
	y[1] = q_curr @ modeOut
	# Damped modal equation: q'' + 2*sig*q' + omega**2 * q = 0
	# with q = exp(s*t) and dividing by exp(s*t)
	# s**2 + 2*sig*s + omega**2 = 0
	# s = -sig +- sqrt(sig**2 -omega**2) = -sig +- 1j*omega_d
	# a1 = 2e^(-sig * dt) * cos(omega_d * dt)
	# a2 = e ^ (-2 * sig * dt)
	# sampled at dt, q[n] = q(n*dt) is a combination of z_+ and z_-
	# Z_+ = exp(s_+ * dt) = exp(-sig*dt) * exp(1j*omega_d*dt)
	# Z_- = exp(s_- * dt) = exp(-sig*dt) * exp(-1j*omega_d*dt)
	# (z-z_+)(z-z_-) = z**2 -(z_+ + z_-)*z + z_+ * z_-
	# so q[n+1] = a1*q[n] - a2*q[n-1] with a1 the sum of roots and a2 the product of roots
	# a1 = z_+ + z_-  = exp(-sig*dt)*(exp(1j*w_d*dt) + exp(-1j*w_d*dt))
	#                 = 2*exp(-sig*dt)*cos(omega_d*dt)
	# a2 = z_+ * z_-  = exp(-sig*dt) * exp(-sig*dt)
	#                 = exp(-2*sig*dt)
	
	a1 = 2 * np.exp(-sig * dt) * np.cos(omega_d * dt)
	a2 = np.exp(-2 * sig * dt)
	for i in range(2,n):
		# q[n+1] = a1 q[n] - a2 q[n-1]
		# At each step, we roll forward previous states 
		q_new = a1 * q_curr - a2 * q_prev
		q_prev = q_curr
		q_curr = q_new
		y[i] = modeOut @ q_curr

		# Tension Modulation
		# A displaced head is stretched so it pulls harder, c = sqrt(T/sigma) and omega = c*k, tension lifts every frequency by gamma
		# S is stretch, weight by k**2, stretch comes from slope and a mode's sloep is its amplitude times its wavenumber
		#   epsilon = S / (2*A_area)          areal strain: fractional extra area
		#   T_eff   = T_0 + Eh*epsilon        Hooke's law for a sheet
		#   gamma   = sqrt(T_eff / T_0)       because omega ∝ sqrt(T)
		#           = sqrt(1 + beta*S)        beta = Eh / (2*A_area*T_0)
		#
		#   S = INT INT |grad u|**2 dA  =  sum_i k_i**2 * q_i**2 * norm_i
		#
		#   omega_i(t) = gamma(t) * omega_i           ONE factor, every mode
		S_tr[i] = kern @ (q_curr**2) # Since modes are eigenfunctions
		# sqrt(1 + beta*S) clamped to 2.0 for numerical reasons
		gamma_c[i] = min(np.sqrt(1 + beta* S_tr[i]), 2.0)
		if i < n_fast:
			L = L_fast # Fast head
		else:
			L = L_slow # Slow Tail
		if i % L == 0: #update 
			g = gamma_c[i]
			omega_d = np.sqrt((g*omega)**2 - sig**2)
			a1 = 2 * np.exp(-sig * dt) * np.cos(omega_d * dt) 
			a2 = np.exp(-2 * sig * dt)
		gamma_a[i] = g

	return (y/np.abs(y).max() if normalize else y), S_tr, gamma_c, gamma_a
'''
def air_load(modes, strength = 1):
	#New mode lsit with frequenices warped toward the measured 26" timpani ratios from Christian et al. 1984
	#Temporary fitting to measured, will model air load later on
	measured = {1:1.00, 2: 1.5, 3:2.00, 4:2.44, 5: 2.9, 6:3.36} # for m = [1,6] only

	j11 = special.jn_zeros(1,1)[0]
	W = {}
	for mm, meas in measured.items():
		ideal = special.jn_zeros(mm,1)[0] / j11
		W[mm] = meas/ideal
	W[0] = 1.0
	W_HIGH = W[6]
	out = []
	for f,m,n,j in modes:
		w_m = W.get(m, W_HIGH)

		f_new = f * (1 + strength * (w_m - 1))
		out.append((f_new, m, n, j))
	out.sort()
	return out 
	'''
def write(path, y, sr):
    wavfile.write(path, int(sr), (np.clip(y, -1, 1) * 32000).astype(np.int16))
'''
md = modes(0.164,100,5000)
print(len(md))

out = strike(md)

md = modes(0.164, 100, 5000)
write('tom.wav',        strike(md),               48000)
write('tom_centre.wav', strike(md, r_strike=0.0), 48000)
'''