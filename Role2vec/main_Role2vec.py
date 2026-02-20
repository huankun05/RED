from .parser import parameter_parser
from .role2vec import Role2Vec
from .utils import tab_printer
import numpy as np
import networkx as nx

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

def rewrite_Role2vec(inpath,outpath,N,dim):
    emb=np.zeros((N, dim), dtype=float)
    infile=open(inpath,'r')
    fistline=infile.readline()
    lines=infile.readlines()
    for line in lines:
        line=(line.strip()).split(',')
        row=int(line[0])-1
        #print(line)
        for i in range(1,dim+1):
            emb[row,i-1]=float(line[i])
        #print(emb)

    writeMatrixTxt(emb.tolist(),outpath)
    print('Role2vec rewrite finishes.')

def main_original(args):
    """
    Role2Vec model fitting.
    :param args: Arguments object.
    """
    #tab_printer(args)
    model = Role2Vec(args)
    model.do_walks()
    model.create_structural_features()
    model.learn_embedding()
    model.save_embedding()


def main_Role2vec(inpath,outpath,dim):
    print('Role2vec starts.')
    G = nx.read_edgelist(inpath)
    N = len((nx.nodes(G)))
    args = parameter_parser(inpath,'embedding_Role2Vec.txt',dim)
    main_original(args)
    rewrite_Role2vec('embedding_Role2Vec.txt', outpath, N, dim)

if __name__ == "__main__":
    main_Role2vec(inpath='topo.txt', outpath='matrix_Role2vec.txt', dim=6)

