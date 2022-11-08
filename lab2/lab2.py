import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp
import math

def Square(x0, y0):
    PX = [x0 - 7.5, x0 - 7.5, x0 + 7.5, x0 + 7.5, x0 - 7.5]
    PY = [y0 - 7.5, y0 + 7.5, y0 + 7.5, y0 - 7.5, y0 - 7.5]
    return PX, PY

def Circle(X, Y, R):
    CX = [X + R * math.cos(i/100) for i in range(0, 628)]
    CY = [Y + R * math.sin(i/100) for i in range(0, 628)]
    return CX, CY

# [Xнач, Xкон], [Yнач, Yкон] => для линии надо сделать конечным фиксированную точку, а начало привязать к вершине блока
def anima(i):
    PrX, PrY = Square(XR[i], YR[i])
    Prism.set_data(PrX, PrY)
    Line_upper.set_data([XR[i] + 0.2, 1], [YR[i] + 7.5, 23.15])
    CX, CY = Circle(XC[i] + 0.3, 1.6 * YC[i] - 4, 3)
    Circle_B.set_data(CX, CY)
    CBX, CBY = Circle(XC[i] - 2.7, YR[i] + 7.3, 0.05)
    Circle_BD.set_data(CBX, CBY)
    Line_bottom.set_data([XC[i] - 2.7, XC[i] - 2.7], [1.6 * YC[i] - 4, YR[i] + 7.5])
    BCX, BCY = Circle(XC[i] + 0.2, 1.6 * YC[i] - 4, 0.05)
    Circle_BС.set_data(BCX, BCY)
    return Prism, Line_upper, Circle_B, Circle_BD, Line_bottom, Circle_BС

t = sp.Symbol('t')
x = 4 * sp.cos(3 * t)
xi = -1.5 * sp.cos(3 * t)

Xr = x * sp.sin(math.pi) + 0.8
Yr = -x * sp.cos(math.pi) + 7.5

Xc = 1.5 * (x * sp.sin(math.pi) + 0.8)
Yc = 2.5 * (xi * sp.cos(math.pi) + 3)

V_X = sp.diff(x, t)
Vx = V_X * sp.cos(math.pi)
Vy = -V_X * sp.sin(math.pi)

Wx = sp.diff(Vx, t)
Wy = sp.diff(Vy, t)

# для графиков
V_Xi = sp.diff(xi, t)
V_X_Xi = V_X + V_Xi
W_X_Xi = sp.diff(V_X_Xi, t)

T = np.linspace(0, 20, 1000)
XR = np.zeros_like(T)
YR = np.zeros_like(T)
XC = np.zeros_like(T)
YC = np.zeros_like(T)
WX = np.zeros_like(T)
WY = np.zeros_like(T)
YX = np.zeros_like(T)
YC = np.zeros_like(T)
VX = np.zeros_like(T)
VY = np.zeros_like(T)

for i in np.arange(len(T)):
    XR[i] = sp.Subs(Xr, t, T[i])
    YR[i] = sp.Subs(Yr, t, T[i])
    XC[i] = sp.Subs(Xc, t, T[i])
    YC[i] = sp.Subs(Yc , t, T[i])
    VX[i] = sp.Subs(Vx, t, T[i])
    VY[i] = sp.Subs(Vy, t, T[i])
    WX[i] = sp.Subs(Wx, t, T[i])
    WY[i] = sp.Subs(Wy, t, T[i])

fig = plt.figure(figsize = (17, 10))

ax1 = fig.add_subplot(121)
ax1.axis('equal')
ax1.set(xlim=[XR.min() - 20, XR.max() + 20], ylim=[YR.min() - 20, YR.max() + 20])

A_R = 1
A_X = 0
A_Y = 22.9
Circle_A = ax1.plot(*Circle(A_X, A_Y, A_R), 'black', linewidth=2)

AD_R = 0.1
AD_X = 0
AD_Y = 22.9
Circle_AD = ax1.plot(*Circle(AD_X, AD_Y, AD_R), 'black', linewidth=3)

B_R = 3
B_X = 0
B_Y = 10
Circle_B = ax1.plot(*Circle(B_X, B_Y, B_R), 'black', linewidth=2)[0]

BD_R = 0.05
BD_X = -2.5
BD_Y = 18.9
Circle_BD = ax1.plot(*Circle(BD_X, BD_Y, BD_R), 'black', linewidth=3)[0]

BС_R = 0.05
BС_X = XR[0] + 0.2
BС_Y = YR[0] + 1.8
Circle_BС = ax1.plot(*Circle(BС_X, BС_Y, BС_R), 'black', linewidth=3)[0]

upper_line_x = [-12, 14]
upper_line_y = [24, 24]
plt.plot(upper_line_x, upper_line_y, 'black')
side_line1_x = [-7.5, -7.5]
side_line1_y = [-10, 24]
plt.plot(side_line1_x, side_line1_y, 'black')
side_line2_x = [9, 9]
side_line2_y = [-10, 24]
plt.plot(side_line2_x, side_line2_y, 'black')

PrX, PrY = Square(XR[0], YR[0])
Prism = ax1.plot(PrX, PrY, 'black')[0]

Line_upper = ax1.plot([1, 1], [22.5, 19], 'black')[0]
Line_bottom = ax1.plot([-1.9, -1.9], [11, 18.8], 'black')[0]


# ГРАФИКИ
T = np.linspace(0, 20, 1000)
VP = np.zeros_like(T)
WP = np.zeros_like(T)
l = np.zeros_like(T)

for i in np.arange(len(T)):
    VP[i] = sp.Subs(V_X_Xi, t, T[i])
    WP[i] = sp.Subs(W_X_Xi, t, T[i])
    l[i] = i

ax2 = fig.add_subplot(424)
ax2.plot(l, VP)
ax2.set_ylabel('V of point')

ax4 = fig.add_subplot(426)
ax4.plot(l, WP)
ax4.set_ylabel('W of point')

plt.subplots_adjust(wspace = 0.3, hspace = 0.7)

anim = FuncAnimation(fig, anima, frames = 1000, interval = 0.01, blit = True)

plt.show()