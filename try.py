import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np


# kernel
# kernel = ['linear', 'poly', 'rbf', 'sigmoid']
# acc_01_= [0.7161, 0.7660, 0.7945, 0.7607]
# acc_001_= [0.6846, 0.4923, 0.7000, 0.6846]


# epoch
epoch=[ 0,      1,       2,      5,     10,     20,     50,     100,    150,    200,    250,    300,    350,    400,    450,     500]
    
lasso_l001_A100 = [0.1959, 0.6291, 0.6637, 0.7398, 0.7611, 0.7724, 0.7772, 0.7776, 0.7788, 0.7792, 0.7794, 0.7793, 0.7795, 0.7796, 0.7797, 0.7800]
lasso_l01_A100 =[0.1959, 0.6291, 0.6637, 0.7398, 0.7611, 0.7724, 0.7771, 0.7776, 0.7788, 0.7792, 0.7794, 0.7792, 0.7795, 0.7796, 0.7797, 0.7799]
lasso_l1_A100  =[0.1959, 0.6291, 0.6637, 0.7399, 0.7612, 0.7722, 0.7771, 0.7776, 0.7785, 0.7792, 0.7791, 0.7792, 0.7797, 0.7797, 0.7798, 0.7797]
lasso_l10_A100 =[0.1959, 0.6294, 0.6641, 0.7408, 0.7615, 0.7726, 0.7762, 0.7779, 0.7784, 0.7789, 0.7790, 0.7793, 0.7797, 0.7800, 0.7798, 0.7800]
lasso_l100_A100=[0.1959, 0.6315, 0.6668, 0.7430, 0.7593, 0.7656, 0.7672, 0.7658, 0.7630, 0.7617, 0.7608, 0.7607, 0.7596, 0.7586, 0.7577, 0.7564]


ridge_l10_A100 =[0.1959, 0.6291, 0.6687, 0.7449, 0.7638, 0.7712, 0.7692, 0.7613]
ridge_l1_A100  =[0.1959, 0.6291, 0.6640, 0.7404, 0.7616, 0.7725, 0.7767, 0.7781, 0.7784, 0.7784, 0.7775, 0.7769, 0.7765, 0.7761, 0.7760, 0.7747]
ridge_l01_A100 =[0.1959, 0.6291, 0.6638, 0.7399, 0.7612, 0.7722, 0.7770, 0.7777, 0.7786, 0.7787, 0.7789, 0.7793, 0.7795, 0.7796, 0.7798, 0.7801]
ridge_l001_A100=[0.1959, 0.6291, 0.6637, 0.7398, 0.7611, 0.7724, 0.7770, 0.7776, 0.7788, 0.7792, 0.7794, 0.7793, 0.7795, 0.7796, 0.7797, 0.7799]
ridge_l05_A10  =[0.1959, 0.6291, 0.6094, 0.6442, 0.6608, 0.6538, 0.7031, 0.6936, 0.6845, 0.7643, 0.7490, 0.7290, 0.7314, 0.7605, 0.6747, 0.6978]

ridge_l05_A100 =[0.1959, 0.6291, 0.6634, 0.7404, 0.7614, 0.7725, 0.7772, 0.7780, 0.7785, 0.7785, 0.7787, 0.7792, 0.7790, 0.7790, 0.7794, 0.7792]


clip_data_01_    =[0.1959, 0.6668, 0.7220, 0.7546, 0.7577, 0.7343, 0.7352, 0.7208, 0.7533, 0.7488, 0.7531, 0.7546, 0.7490, 0.7562, 0.7504, 0.7547]
scale_data_01_    =[0.1959, 0.7526, 0.7328, 0.7527, 0.7632, 0.7722, 0.7773, 0.7778, 0.7788, 0.7794, 0.7793, 0.7792, 0.7794, 0.7798, 0.7799, 0.7799]
data_1_b   =[0.1959, 0.7105, 0.7323, 0.7517, 0.7631, 0.7706, 0.7751, 0.7752, 0.7757, 0.7763, 0.7765, 0.7767, 0.7768, 0.7769, 0.7769, 0.7770] 
data_01_b  =[0.1952, 0.6223, 0.6331, 0.6715, 0.7018, 0.7226, 0.7456, 0.7564, 0.7618, 0.7637, 0.7637, 0.7653, 0.7664, 0.7660, 0.7649, 0.7641]
data_001_b =[0.2269, 0.5846, 0.6846, 0.7154, 0.7115, 0.7154, 0.7077, 0.7038, 0.7038, 0.7038, 0.7077]
        



plt.plot()


plt.xlabel("epoch")
plt.ylabel("Accuracy")

t = 7
# plt.plot(epoch[:t], lasso_l01_A100[:t], label='RL + Lasso',linewidth=1.0)
# plt.plot(epoch[:t], ridge_l10_A100[:t], label='RL + Ridge',linewidth=1.0)
# plt.plot(epoch[:t], scale_data_01_[:t], label='RL',linewidth=1.0)
# plt.plot(epoch[:t], clip_data_01_[:t], label='RL(CLAP)',linewidth=1.0, linestyle=':')
# plt.plot(epoch[:t], data_1_b[:t], label='RL(-b)',linewidth=1.0,linestyle='-.')
# plt.plot(epoch[:t], ridge_l05_A10[:t], label='RL(A/10)',linewidth=1.0,linestyle='--')
plt.plot(epoch[:t], scale_data_01_[:t], label='whole data(+b)',linewidth=1.0)
plt.plot(epoch[:t], data_1_b[:t], label='whole data',linewidth=1.0)
plt.plot(epoch[:t], data_01_b[:t], label='1/10 data',linewidth=1.0)
plt.plot(epoch[:t], data_001_b[:t], label='1/100 data',linewidth=1.0)
plt.legend(loc='best')
plt.show()


x = np.linspace(-3,3,50)
y1 = 2*x - 1
y2 = x**2

plt.figure()
plt.plot(x,y1,label='Linear function')
plt.plot(x,y2,label='Quadratic function',color="red",linewidth=1.0,linestyle='--')

# 限制x,y轴的范围，设置标签
plt.xlim((-2,3))
plt.ylim((-2,8))
plt.xlabel("x")
plt.ylabel("y")

# 更换下标
new_ticks = np.linspace(-2,3,10)
plt.xticks(new_ticks)
plt.yticks(range(5), ['cat', 'fish', 'dog$', 'tom', 'jerry'])

# 移动x,y轴位置
# gca = "get current axis"
ax = plt.gca()
ax.spines['right'].set_color('none') # 右边框设置成无颜色
ax.spines['top'].set_color('none') # 上边框设置成无颜色
ax.xaxis.set_ticks_position('bottom') # x轴用下边框代替，默认是这样
ax.yaxis.set_ticks_position('left') # y轴用左边的边框代替，默认是这样
ax.spines['bottom'].set_position(('data',0)) # x轴在y轴，０的位置
ax.spines['left'].set_position(('data',0)) # y轴在x轴，０的位置


# annotation 注释，我们注释在交点
# emm,我手算出来，暂时没考虑样自动计算
# 这个略微有点复杂，用的时候，google下就好
x0 = 1
y0 = 2*x0 -1
plt.scatter(x0,y0,color='green') # 画一个点
plt.plot([x0,x0],[y0,0],color='green',linestyle='--') # 画一条虚线
plt.annotate('intersection is (%d,%d)' % (x0,y0),
                xy=(x0,y0),xytext=(x0+0.5,y0-0.5),xycoords='data',
                arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))

# plt.text(0,-2,'unused text.')

# 设置透明度
# for label in ax.get_xticklabels() + ax.get_yticklabels():
#     # label.set_fontsize(16)
#     label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.65 ))

plt.legend(loc='best')
# plt.show()


 # acc=0.5186, 0.4308
        # acc=0.5186 , 0.23846
        #  



    