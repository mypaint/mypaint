from pylab import *

dab = imread('parametric_dab.png')

print dab.shape

l = dab[500, :, 3]
#plot(l, label="opacity")

r = hstack((linspace(1, 0, 500), linspace(0, 1, 500)[1:]))
#plot(r, label="$r$")

rr = r**2 
o = 1.0-rr
#plot(o, label="$1-r^2$")

for i in [1, 2]:
    figure(i)
    hardness = 0.3
    for hardness in [0.1, 0.3, 0.7]:
        if i == 2:
            rr = linspace(0, 1, 1000)
        opa = rr.copy()
        opa[rr<hardness] = rr[rr<hardness] + 1-(rr[rr<hardness]/hardness)
        opa[rr>=hardness] = hardness/(1-hardness)*(1-rr[rr>=hardness])
        plot(opa, label="h=%.1f" % hardness)
        if i == 2:
            xlabel("$r^2$")
            legend(loc='best')
        else:
            xlabel("$d$")
            legend(loc='lower center')

    ylabel('pixel opacity')
    xticks([500], [0])
    title("Dab Shape (for different hardness values)")

show()
