import numpy as np
np.random.seed(2)
a=np.random.rand(3,3)
print(a.ndim)#ndim输出维度，shape输出元祖
'''with open(csv.path,'r') as fh:  #数据集处理方法，读标题，根据标题字符串匹配区分列·
    header=fh.readline().strip().split(',')   #[x1,x2,y]
    x_col=[i for i in range(len(header)) if header[i].startswith('x')]
    y_col=[i for i in range(len(header)) if header[i]=='y']
    inputvector=np.loadtxt()#从文本中提取数组,参数,文件名,dtype,delimiter分隔符,skiprows跳过的行数,usecols指定列的序号
    #a=np.expand_dims(,axis=)#axis0,1,-1...指定shape元祖加维度的位置,-1为最后
'''




    