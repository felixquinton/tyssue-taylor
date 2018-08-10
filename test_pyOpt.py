import pyOpt



def objfunc(x):

  f = x-2
  g = [0.0]
  g[0] = -x - 4
  fail = 0
  return f,g, fail

opt_prob = pyOpt.Optimization('TP37 Constrained Problem',objfunc)

opt_prob.addObj('f')

opt_prob.addVar('x1','c',lower=0.0,upper=42.0,value=10.0)
opt_prob.addCon('g','i')
print(opt_prob)
slsqp = pyOpt.PSQP()
slsqp.setOption('IPRINT', -1)
[fstr, xstr, inform] = slsqp(opt_prob,sens_type='FD')
print (opt_prob.solution(0))
