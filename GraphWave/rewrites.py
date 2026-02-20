import  numpy as np
import csv

# 接收矩阵文件的路径，读取之
def readMatrix(filepath,type):
    infile = open(filepath,'r')
    lines = infile.readlines()
    rows = len(lines)  # 行数
    cols=len(lines[0].strip().split())  # 列数
    #print('Row='+str(rows)+' '+'Cols='+str(cols))
    A = np.zeros((rows, cols), dtype=type)
    A_row=0
    for line in lines:
        line = (line.strip()).split()
        A[A_row:] = line
        A_row += 1
    infile.close()
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

# 【GraphWave的预处理输出】
def writeCsvGW(topopath,outpath='topoedges.csv'):
    matrix=readMatrix(topopath,int)
    # 解析embeddingpath拿到当前算法名，构建输出文件名
    headers = ['node_1', 'node_2']
    f=open(outpath, 'w', newline='')
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    f_csv.writerows(matrix-1)
    f.close()
    return 0

def rewrite(outpath,dim,N):
    csvIn=open("embedding.csv",'r')
    csvIn.readline()
    lines=csvIn.readlines()
    emb = np.zeros((N, dim), dtype=complex)
    emb2 = np.zeros((N, 2*dim), dtype=float)
    for line in lines:
        line=(line.strip()).split(',')
        #print(line)
        index=int(line[0])
        for i in range(0,dim):
            #print(float(line[i+1]))
            c=complex(float(line[i+1]),float(line[i+1+dim]))
            emb[index,i]=c
            emb2[index,i]=float(line[i+1])
            emb2[index,i+dim]=float(line[i+1+dim])
    #print(emb)
    writeMatrixTxt(emb.tolist(),outpath[:-4]+'C.txt')
    writeMatrixTxt(emb2.tolist(), outpath)
    print('GraphWave rewrite finishes.')
    return 0


if __name__ == "__main__":
    writeCsvGW('topo.txt')
