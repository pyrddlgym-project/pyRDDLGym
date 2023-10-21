import matplotlib.pyplot as plt
# seed = 3264

res = {
3264: """N= 9.0 Best return= -21609.238452414433
N= 10.0 Best return= -21609.238452414433
N= 7.0 Best return= -22337.746159785675
N= 5.0 Best return= -26668.44143948945
N= 0.0 Best return= -24886.181083912685
N= 1.0 Best return= -24445.63925966282
N= 4.0 Best return= -25179.912454570454
N= 6.0 Best return= -24629.414291395966
N= 8.0 Best return= -21609.238452414433
N= 3.0 Best return= -24917.986075335437
N= 2.0 Best return= -24863.596535065823""",
42: """N= 2.0 Best return= -25089.89395452745
N= 1.0 Best return= -25082.695177696332
N= 8.0 Best return= -21562.387786382686
N= 4.0 Best return= -25092.452438277942
N= 3.0 Best return= -25585.84587201423
N= 9.0 Best return= -21609.238452414433
N= 0.0 Best return= -25256.74085443676
N= 5.0 Best return= -24760.572247327436
N= 7.0 Best return= -24347.90832413362
N= 10.0 Best return= -21609.238452414433
N= 6.0 Best return= -24164.551004786175""",
1686: """N= 10.0 Best return= -21609.238452414433
N= 3.0 Best return= -24828.139627361343
N= 0.0 Best return= -25342.726900322166
N= 4.0 Best return= -24909.378050408464
N= 8.0 Best return= -21609.238452414433
N= 6.0 Best return= -25236.080059198288
N= 1.0 Best return= -25047.881850147456
N= 9.0 Best return= -21609.238452414433
N= 7.0 Best return= -22150.90795972415
N= 2.0 Best return= -24744.972206488026
N= 5.0 Best return= -25009.79286870775""",
2220: """N= 8.0 Best return= -21609.238452414433
N= 6.0 Best return= -24015.356694031267
N= 5.0 Best return= -24166.65862343653
N= 0.0 Best return= -25341.91457204941
N= 4.0 Best return= -24491.03298888002
N= 3.0 Best return= -24954.436526570255
N= 9.0 Best return= -21609.238452414433
N= 10.0 Best return= -21609.238452414433
N= 1.0 Best return= -24508.09422272449
N= 7.0 Best return= -21537.94807416154
N= 2.0 Best return= -25070.326875770494""",
9285: """N= 2.0 Best return= -24135.55045189298
N= 4.0 Best return= -24144.47213610023
N= 1.0 Best return= -24325.63583798047
N= 10.0 Best return= -21609.238452414433
N= 7.0 Best return= -24101.810480627464
N= 0.0 Best return= -24353.93032824988
N= 9.0 Best return= -21609.238452414433
N= 5.0 Best return= -24407.272563828326
N= 6.0 Best return= -24796.76025227404
N= 3.0 Best return= -24503.672112988606""",
}

res2 = {
'w02': """N= 10.0 Best return= -21609.238452414433
N= 1.0 Best return= -24586.095311729263
N= 8.0 Best return= -21609.238452414433
N= 3.0 Best return= -24709.204784095215
N= 5.0 Best return= -24730.748175826768
N= 9.0 Best return= -21609.238452414433
N= 7.0 Best return= -25008.753293223082
N= 6.0 Best return= -25957.33822666952
N= 0.0 Best return= -25303.10933352667
N= 4.0 Best return= -24705.169049992965
N= 2.0 Best return= -24484.347816053774""",
'w03': """N= 6.0 Best return= -25957.33822666952
N= 10.0 Best return= -21609.238452414433
N= 4.0 Best return= -24705.169049992965
N= 2.0 Best return= -24490.4041965704
N= 7.0 Best return= -25008.753293223082
N= 3.0 Best return= -24709.204784095215
N= 5.0 Best return= -24733.83340675022
N= 0.0 Best return= -25303.10933352667
N= 9.0 Best return= -21609.238452414433
N= 8.0 Best return= -21609.238452414433
N= 1.0 Best return= -24534.26823748197""",
'w05': """N= 9.0 Best return= -21609.238452414433
N= 10.0 Best return= -21609.238452414433
N= 8.0 Best return= -21609.238452414433
N= 3.0 Best return= -24709.204784095215
N= 7.0 Best return= -25008.753293223082
N= 4.0 Best return= -24705.169049992965
N= 2.0 Best return= -24487.79332840509
N= 1.0 Best return= -24589.687415683977
N= 0.0 Best return= -25249.655217540017
N= 6.0 Best return= -25504.033073639755
N= 5.0 Best return= -24733.83340675022""",
'w10': """N= 0.0 Best return= -24571.848134576892
N= 2.0 Best return= -24423.640226418123
N= 8.0 Best return= -21609.238452414433
N= 4.0 Best return= -24705.169049992965
N= 1.0 Best return= -24490.281734693523
N= 7.0 Best return= -23368.542497275754
N= 10.0 Best return= -21609.238452414433
N= 6.0 Best return= -25696.635065217706
N= 5.0 Best return= -24502.58521558887
N= 9.0 Best return= -21609.238452414433
N= 3.0 Best return= -24659.49542562674""",
'w15': """N= 9.0 Best return= -21609.238452414433
N= 10.0 Best return= -21609.238452414433
N= 7.0 Best return= -22337.746159785675
N= 5.0 Best return= -26668.44143948945
N= 0.0 Best return= -24886.181083912685
N= 1.0 Best return= -24445.63925966282
N= 4.0 Best return= -25179.912454570454
N= 6.0 Best return= -24629.414291395966
N= 8.0 Best return= -21609.238452414433
N= 3.0 Best return= -24917.986075335437
N= 2.0 Best return= -24863.596535065823""",
'w20': """N= 6.0 Best return= -25705.16926797406
N= 4.0 Best return= -24460.139283962344
N= 7.0 Best return= -22788.84801437964
N= 3.0 Best return= -23995.084421330575
N= 10.0 Best return= -21609.238452414433
N= 1.0 Best return= -24222.510796107686
N= 0.0 Best return= -24543.159872897013
N= 8.0 Best return= -21609.238452414433
N= 9.0 Best return= -21609.238452414433
N= 2.0 Best return= -24523.059847529134
N= 5.0 Best return= -24421.24552597492""",
'w25': """N= 8.0 Best return= -21609.238452414433
N= 10.0 Best return= -21609.238452414433
N= 6.0 Best return= -25911.838000346397
N= 9.0 Best return= -21609.238452414433
N= 7.0 Best return= -24971.383483362428
N= 3.0 Best return= -24296.779511016583
N= 5.0 Best return= -24733.833406750222
N= 1.0 Best return= -24589.687415683977
N= 2.0 Best return= -24257.646427786352
N= 0.0 Best return= -24433.70090036745
N= 4.0 Best return= -24699.11887192531""",
#'42w10': """N= 7.0 Best return= -22617.589566604405
#N= 2.0 Best return= -25244.97997906033
#N= 9.0 Best return= -21539.339558304862
#N= 6.0 Best return= -24894.89950705054
#N= 4.0 Best return= -24879.78938320209
#N= 10.0 Best return= -21609.238452414433
#N= 3.0 Best return= -25270.18319295039
#N= 0.0 Best return= -25382.300208303684
#N= 1.0 Best return= -25216.234556243355
#N= 8.0 Best return= -21515.23104947676
#N= 5.0 Best return= -24608.56350463354""",
}

fig, ax = plt.subplots(1,2)
fig.set_size_inches(14,7)

for seed, r in res.items():
    res_tuples = []
    for line in r.split('\n'):
        s = line.split('=')
        N, ret = s[1], s[2]
        N = N.strip(' ').split(' ')[0]
        res_tuples.append((float(N)/10, float(ret)))

    res_tuples = tuple(sorted(res_tuples, key=lambda x: x[0]))
    Ns = [x[0] for x in res_tuples]
    rs = [x[1] for x in res_tuples]

    ax[0].plot(Ns, rs, label=seed)

for seed, r in res2.items():
    res_tuples = []
    for line in r.split('\n'):
        s = line.split('=')
        N, ret = s[1], s[2]
        N = N.strip(' ').split(' ')[0]
        res_tuples.append((float(N)/10, float(ret)))

    res_tuples = tuple(sorted(res_tuples, key=lambda x: x[0]))
    Ns = [x[0] for x in res_tuples]
    rs = [x[1] for x in res_tuples]

    ax[1].plot(Ns, rs, label=seed)

for i in range(2):
    ax[i].plot(Ns, [-25229]*len(Ns), label='Min', linestyle='dashed')
    ax[i].plot(Ns, [-25386]*len(Ns), label='Max', linestyle='dashed')
    ax[i].plot(Ns, [-21652]*len(Ns), label='Offset', linestyle='dashed')


ax[0].set_title('w=15')
ax[0].set_xlabel('lambda')
ax[0].set_ylabel('-Total travel time (s)')
ax[1].set_title('seed=3264')
ax[1].set_xlabel('lambda')
ax[0].legend()
ax[0].grid(visible=True)
ax[1].legend()
ax[1].grid(visible=True)

plt.suptitle('Green wave. 5 intersections. Best performance in 3000 iterations\nInitial plan = lambda*offset + (1-lambda)*random')
plt.tight_layout()
plt.show()
