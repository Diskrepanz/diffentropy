from sympy import integrate, symbols, log

# if 0 <= x < 0.25:
#     return float(0)
# elif 0.25 <= x < 0.5:
#     return 16.0 * (x - 0.25)
# elif 0.5 <= x < 0.75:
#     return -16.0 * (x - 0.75)
# elif 0.75 < x <= 1:
#     return float(0)
# h(f) = integrate(-f(x)lnf(x), (x, 0, 1))

x = symbols('x')
left = integrate(-16.0 * (x - 0.25) * log(16.0 * (x - 0.25)), (x, 0.25, 0.5))
right = integrate(16.0 * (x - 0.75) * log(-16.0 * (x - 0.75)), (x, 0.5, 0.75))
print 'left {0}'.format(left)
print 'right {0}'.format(right)
print 'all {0}'.format(left + right)
