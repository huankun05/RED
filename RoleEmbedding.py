from numpy import *  #导入numpy的库函数
import numpy as np   #这个方式使用numpy的函数时，需要以np.开头。
import diameter
import math
import time

#输入EL输出AM
def ReadEL(filein):
    fin = open(filein,'r')
    #N=number of nodes
    N=0
    f_lines = fin.readlines()
    for line in f_lines:
        line = line.strip()
        vec = line.split()
        if (int(vec[0]) > N):
            N=int(vec[0])
        if (int(vec[1]) > N):
            N=int(vec[1])
    fin.close()
    A = np.zeros((N, N), dtype=int)
    fin = open(filein,'r')
    f_lines = fin.readlines()
    for line in f_lines:
        line = line.strip()
        vec = line.split()
        a,b=int(vec[0])-1,int(vec[1])-1
        A[a][b]=1
        A[b][a]=1
    fin.close()
    return A


#输出list形式的矩阵（array可以tolist()），到txt中
def writeMatrixTxt(matrixlist,filepath):
    outfile=open(filepath,'w')
    for line in matrixlist:
        linestr=''
        if type(line)!=list:
            #print(line,'column=1')
            linestr += str(line)  # for column vector
        else:
            for i in line:
                linestr+=str(i)+'\t'
            linestr=linestr[:-1]+'\n'
        outfile.write(linestr)
    outfile.close()


# 修复后的偏移DTQW
# biased DTQW
# A=邻接矩阵，steps=步数，starting=起点，starting_ratio=概率比重
# 边的振幅矩阵matrix，节点的概率向量probvec【边概率必须横向累加得到点概率！！！！！！！！！！！！！】
# 每一步都输出probvec，合并成矩阵，第i行为第i步的probvec
#【更改】累加，得到总的probvec
def DTQW_biased(A,steps,starting,starting_ratio=1):
    N=shape(A)[1] #节点数
    D=reshape((A.sum(axis=1)),(N,1)) #度数列向量
    # 使用np.asmatrix替代mat，以适配NumPy 2.0
    DD=np.asmatrix(np.tile(D,N)) #度数列向量横向复制的矩阵

    # 除偏向点外，剩余概率对剩余点均分，偏向点独享大概率，然后所有点在点内按边均分
    T0 = A * ((1 - starting_ratio) / (N - 1))
    T0[starting, :] = A[starting,:] * (starting_ratio)
    for i in range(N):
        T0[i, :] = T0[i, :] / (D[i])
    # 使用np.asmatrix替代mat，以适配NumPy 2.0
    T0 = np.asmatrix(sqrt(T0))

    # 使用np.asmatrix替代mat，以适配NumPy 2.0
    probvecMatrix=np.asmatrix(np.zeros(shape=(steps,N))) #概率汇总矩阵，行数=steps，列数=N，第i行为第i步的probvec
    for i in range(steps):
        T1 = multiply(divide(2.0,DD)-1, T0.T)
        T0d = multiply(divide(2.0,DD), T0.T)
        T1 = T1 + multiply( np.tile(T0d.sum(axis=1),N), A) - T0d
        T0 = T1

        probability_matrix = multiply(abs(T0), abs(T0))  # 各边概率的矩阵
        '''probvec = probability_matrix.sum(axis=0)  # 各点概率的行向量
        probvecMatrix[i, :] = probvec

    #return probvecMatrix
    return probvecMatrix.sum(axis=0)'''
        #【边概率必须横向累加得到点概率！！！！！！！！！！！！！】
        probvec = probability_matrix.sum(axis=1)  # 各点概率的列向量
        probvecMatrix[i, :] = probvec.T
    return probvecMatrix.sum(axis=0)

# 修复后的DTQW
# DTQW，A=邻接矩阵，steps=步数
# 边的振幅矩阵matrix，节点的概率向量probvec【边概率必须横向累加得到点概率！！！！！！！！！！！！！】
# 每一步都输出probvec，合并成矩阵，第i行为第i步的probvec
# 输出的矩阵进行转置，即每行表示一个点，每列表示一步
def DTQW(A,steps):
    N=shape(A)[1] #节点数
    D=reshape((A.sum(axis=1)),(N,1)) #度数列向量
    # 使用np.asmatrix替代mat，以适配NumPy 2.0
    DD=np.asmatrix(np.tile(D,N)) #度数列向量横向复制的矩阵
    #sumt0=D.sum() #度数和，亦即2|E|
    # 【二重均匀的均匀初态】
    T0 = A * (1/(N))
    for i in range(N):
        T0[i, :] = T0[i, :]/(D[i])
    # 使用np.asmatrix替代mat，以适配NumPy 2.0
    T0 = np.asmatrix(sqrt(T0))

    # 使用np.asmatrix替代mat，以适配NumPy 2.0
    probvecMatrix=np.asmatrix(np.zeros(shape=(steps,N))) #概率汇总矩阵，行数=steps，列数=N，第i行为第i步的probvec
    for i in range(steps):
        T1 = multiply(divide(2.0,DD)-1, T0.T)
        T0d = multiply(divide(2.0,DD), T0.T)
        T1 = T1 + multiply( np.tile(T0d.sum(axis=1),N), A) - T0d
        T0 = T1

        probability_matrix = multiply(abs(T0), abs(T0))  # 各边概率的矩阵
        #probvec = probability_matrix.sum(axis=0)  # 各点概率的行向量
        #probvecMatrix[i, :] = probvec
        # 【边概率必须横向累加得到点概率！！！！！！！！！！！！！】
        probvec = probability_matrix.sum(axis=1)  # 各点概率的列向量
        probvecMatrix[i, :] = probvec.T

    return probvecMatrix.T


# RED: Role Embedding based on DTQW
# inpath: input graph file, Edgelist form
# outpath: output embedding path
# dim: length of embeddings, namely 'd' in our paper
# d1: length of global representations, namely 'd_g' in our paper
# D: diameter of G
def RoleEmbeddingD(inpath, outpath, dim, d1):
    if d1>dim:
        print('d1 should <= dim')
        exit(1)
    A = ReadEL(inpath)
    N = shape(A)[1]
    dia=diameter.diameter(inpath)

    # global / sequences
    history=DTQW(A,d1)


    # closeness vec
    proximityMatrix = np.zeros(shape=(N, N))
    for i in range(N):
        probvec = DTQW_biased(A, dia, i)
        proximityMatrix[i, :] = array((probvec))

    # local
    d2=dim-d1
    roleFeature=-np.sort(-proximityMatrix)[:,:d2]

    emb=np.hstack((history, roleFeature))
    writeMatrixTxt(emb.tolist(), outpath+ '.txt')

    print('RED finishes')
    return [outpath + '.txt']

if __name__ == "__main__":
    RoleEmbeddingD(inpath='topo.txt', outpath='matrix_', dim=6,d1=3)
