import warnings
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sys

# 首先确保能够找到pandas等依赖库
# 然后添加项目中的GraphRole库路径到最前面，确保优先使用项目中的版本
sys.path.insert(0, '../GraphRole')

# 现在导入项目中的GraphRole库中的所有模块
from graphrole import RecursiveFeatureExtractor, RoleExtractor
#import seaborn as sns

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

def rewrite_rolx(inpath,outpath,N,dim):
    emb=np.zeros((N, dim), dtype=float)
    infile=open(inpath,'r')
    #fistline=infile.readline()
    lines=infile.readlines()
    for line in lines:
        line=(line.strip()).split(',')
        row=int(line[0])-1
        #print(line)
        for i in range(1,dim+1):
            emb[row,i-1]=float(line[i])
        #print(emb)

    writeMatrixTxt(emb.tolist(),outpath)
    print('Rolx rewrite finishes.\n')


def main_rolx(inpath,outpath,dim):
    print('Rolx starts.')
    # load the well known karate_club_graph from Networkx
    #G = nx.karate_club_graph()
    G=nx.read_edgelist(inpath)
    N=len((nx.nodes(G)))
    #print(nx.nodes(G))

    # 已经在文件开头导入了RecursiveFeatureExtractor和RoleExtractor

    # extract features
    feature_extractor = RecursiveFeatureExtractor(G)
    features = feature_extractor.extract_features()
    #print('Rolx features ends.')
    #print(f'\nFeatures extracted from {feature_extractor.generation_count} recursive generations:')
    #print(features)

    # assign node roles
    role_extractor = RoleExtractor(n_roles=dim) #######################roles=dimensions
    role_extractor.extract_role_factors(features)
    node_roles = role_extractor.roles
    #print('Rolx roles ends')
    #print('\nNode role assignments:')
    #print(node_roles)
    #print('\nNode role membership by percentage:')
    #print(role_extractor.role_percentage.round(2))

    role_extractor.role_percentage.to_csv( 'embedding_rolx.txt',sep=',', header=False, index=True)
    rewrite_rolx('embedding_rolx.txt',outpath,N,dim)

    # build color palette for plotting
    '''
    unique_roles = sorted(set(node_roles.values()))
    color_map = sns.color_palette('Paired', n_colors=len(unique_roles))
    # map roles to colors
    role_colors = {role: color_map[i] for i, role in enumerate(unique_roles)}
    # build list of colors for all nodes in G
    node_colors = [role_colors[node_roles[node]] for node in G.nodes]

    # plot graph
    plt.figure()
    with warnings.catch_warnings():
        # catch matplotlib deprecation warning
        warnings.simplefilter('ignore')
        nx.draw(
            G,
            pos=nx.spring_layout(G, seed=42),
            with_labels=True,
            node_color=node_colors,
        )
    plt.show()'''

if __name__ == '__main__':
    main_rolx(inpath='topo.txt',outpath='matrix_rolx.txt',dim=2)