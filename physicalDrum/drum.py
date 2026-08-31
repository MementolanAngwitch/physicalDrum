from scipy import special
import numpy as np 
from scipy.io import wavfile
import matplotlib.pyplot as plt 

sr = 48000

#Circular surface, bessel function
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
		n_zeros = int(2*a*f_max/c) + 8
		zeros = special.jn_zeros(m,64)
		f = c * zeros / (2 * np.pi *a)
		g = f < f_max
		if not g.any():
			break
		zeros = zeros[:g.sum()]

		if m == 0: eps = 2.0  
		else: eps = 1.0

		for i,z in enumerate(zeros):
			j = z
			k = j/a 
			modeIn = special.jv(m, j*r_strike) 
			modeOut = special.jv(m, j*r_pickup) * np.cos(theta*m) # Pick theta = 0 for strike point
		
			norm = eps * (np.pi * (a ** 2)/2 ) * special.jv(m+1, j) ** 2
			out.append((k,f[i],modeIn,modeOut,norm))
		m += 1
	out.sort(key=lambda r: r[1])
	props = {'A_area': np.pi *a ** 2, 'T_0': T_0, 'sigma_mu': rho*h, 'c':c, 'h':h}
	#print(c)
	#print(out[0][1])
	return out, props


#Rectangular surface, sin and cos
def rect_modes(Lx=0.356012, Ly=0.237341, f_max=5000,
               x_strike=0.618, y_strike=0.382, x_pickup=0.276, y_pickup=0.723,
               T_0=3530.6, rho=1390, h=2.54e-4):
	c = np.sqrt(T_0/(rho*h))
	out = []
	m = 1  
	while True:
		n = 1
		any_this_m = False 
		while True:
			k = np.pi * np.hypot(m/Lx, n/Ly) # geometry, no bessel zeros
			f = c*k / (2*np.pi)
			if f >= f_max: 
				break
			modeIn = np.sin(m*np.pi*x_strike) * np.sin(n*np.pi*y_strike)
			modeOut = np.sin(m*np.pi*x_pickup) * np.sin(n*np.pi*y_pickup)
			norm = Lx*Ly / 4.0 
			out.append((k,f,modeIn,modeOut,norm))
			any_this_m = True
			n += 1 
		if not any_this_m: break
		m += 1
	out.sort(key=lambda r: r[1])
	props = {'A_area': Lx*Ly, 'T_0': T_0, 'sigma_mu': rho*h, 'c': c, 'h': h}
	return out, props

def sequential_strike(table, props, dur = 2.0 , sr= 48000, w = 0.00984, sigma_a=2.0, sigma_b=3.723e-5, 
	P=1.73e-2, E = 4.895e9, normalize = True): #want to remove all 
	#  sigma_a/_b are damping elements, E is young's modulus
	# sigma_mu is areal density and P is strike impulse
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

def write(path, y, sr):
    wavfile.write(path, int(sr), (np.clip(y, -1, 1) * 32000).astype(np.int16))
