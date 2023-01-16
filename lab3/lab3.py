import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
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

# defining
m = 1
M = 100
c = 100
a = 0.1
t0 = 0
xi0 = 0.1
dxi0dt = 0.1
g = 9.81
R = 1

t = sp.Symbol('t')
x = sp.Function('x')(t)
xi = sp.Function('xi')(t)
xH = sp.Function('xH')(t)
xiH = sp.Function('xiH')(t)

Jc = (m * R * R) / 2
w = xiH / R
T1 = (M * xH * xH) / 2
T2 = (m * (xH + xiH) * (xH + xiH)) / 2 + (Jc * w * w) / 2

Tc = T1 + sp.simplify(T2)

Pc = -M * g * x - m * g * (x + xi) + (c * ((x - a) * (x - a))) / 2

Lc = Tc - Pc

ur1 = sp.diff(sp.diff(Lc, xH), t) - sp.diff(Lc, x)
ur2 = sp.diff(sp.diff(Lc, xiH), t) - sp.diff(Lc, xi) - g

print(ur1)
print(ur2)

a11 = ur1.coeff(sp.diff(xH, t), 1)
a12 = ur1.coeff(sp.diff(xiH, t), 1)
a21 = ur2.coeff(sp.diff(xH, t), 1)
a22 = ur2.coeff(sp.diff(xiH, t), 1)

b1 = -(ur1.coeff(sp.diff(xH, t), 0)).coeff(sp.diff(xiH, t), 0).subs([(sp.diff(x, t), xH), (sp.diff(xi, t), xiH)])
b2 = -(ur2.coeff(sp.diff(xH, t), 0)).coeff(sp.diff(xiH, t), 0).subs([(sp.diff(x, t), xH), (sp.diff(xi, t), xiH)])

detA = a11 * a22 - a12 * a21
detA1 = b1 * a22 - b2 * a12
detA2 = a11 * b2 - b1 * a21

dxHdt = detA1 / detA
dxiHdt = detA2 / detA


def formY(y, t, fV, fOm):
    y1, y2, y3, y4 = y
    dydt = [y3, y4, fV(y1, y2, y3, y4), fOm(y1, y2, y3, y4)]
    return dydt

countOfFrames = 500
T = np.linspace(0, 2, countOfFrames)

fxH = sp.lambdify([x, xi, xH, xiH], dxHdt, "numpy")
fxiH = sp.lambdify([x, xi, xH, xiH], dxiHdt, "numpy")
y0 = [0.1, xi0, dxi0dt, dxi0dt]
sol = odeint(formY, y0, T, args = (fxH, fxiH))

x = sol[:,0]
xi = sol[:,1]
xH = sol[:,2]
xiH = sol[:,3]

XR = [0] * len(x)
YR = [0] * len(x)
XC = [0] * len(x)
YC = [0] * len(x)
l = [0] * len(x)
VP = [0] * len(x)
WP = [0] * len(x)

for i in range(len(x)):
    XR[i] = 0.8
    YR[i] = 3 * (sp.cos(x[i]) + sp.cos(xi[i])) + 7.5
    XC[i] = 1.5 * 0.8
    YC[i] = 2.5 * (sp.cos(xi[i])) + 8
    VP[i] = 3 * (sp.cos(x[i]) + sp.cos(xi[i])) + 2.5 * sp.cos(xi[i])
    l[i] = i

fig = plt.figure(figsize = (17, 10))
ax1 = fig.add_subplot(121)
ax1.axis('equal')
ax1.set(xlim=[x.min() - 20, x.max() + 20], ylim=[19, 21])

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
BС_X = x[0] + 0.2
BС_Y = x[0] + 1.8
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

PrX, PrY = Square(x[0], x[0])
Prism = ax1.plot(PrX, PrY, 'black')[0]

Line_upper = ax1.plot([1, 1], [22.5, 19], 'black')[0]
Line_bottom = ax1.plot([-1.9, -1.9], [11, 18.8], 'black')[0]

ax2 = fig.add_subplot(424)
ax2.plot(l, VP)
ax2.set_ylabel('V of point')

plt.subplots_adjust(wspace = 0.3, hspace = 0.7)

anim = FuncAnimation(fig, anima, frames = 320, interval = 0.01, blit = True)

plt.show()
