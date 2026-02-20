import numpy as np

# 【EL读取】接收EL路径，读取并返回AM
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

def Degree(inpath,outpath,dim):
    print("Degree Starts")
    A=ReadEL(inpath)
    N = np.shape(A)[1]  # 节点数
    D = np.reshape((A.sum(axis=1)), (N, 1))  # 度数列向量
    # 使用np.asmatrix替代np.mat，以适配NumPy 2.0
    DD = np.asmatrix(np.tile(D, dim))  # 度数列向量横向复制的矩阵
    #print(DD)
    writeMatrixTxt(DD.tolist(), outpath)

    print("Degree Ends")

if __name__ == "__main__":
    Degree(inpath='topo.txt', outpath='matrix_Degree.txt', dim=6)
