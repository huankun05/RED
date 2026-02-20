import numpy as np
from numpy import * 
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
from sklearn import metrics
import warnings
import os

warnings.filterwarnings('ignore')

def ReadEL(filein):
    fin = open(filein,'r')
    N=0
    f_lines = fin.readlines()
    for line in f_lines:
        line = line.strip()
        if not line: continue
        vec = line.split()
        if (int(vec[0]) > N): N=int(vec[0])
        if (int(vec[1]) > N): N=int(vec[1])
    fin.close()
    A = np.zeros((N, N), dtype=int)
    fin = open(filein,'r')
    for line in fin:
        line = line.strip()
        if not line: continue
        vec = line.split()
        a,b=int(vec[0])-1,int(vec[1])-1
        A[a][b]=1; A[b][a]=1
    fin.close()
    return A

def readMatrix(filepath, type):
    with open(filepath, 'r') as infile:
        lines = infile.readlines()
        rows = len(lines)
        cols = len(lines[0].strip().split())
        A = np.zeros((rows, cols), dtype=type)
        for i, line in enumerate(lines):
            A[i, :] = line.strip().split()
    return A

def writeMatrixTxt(matrixlist, filepath):
    with open(filepath, 'w') as outfile:
        for line in matrixlist:
            if isinstance(line, (list, np.ndarray)):
                outfile.write('\t'.join(map(str, line)) + '\n')
            else:
                outfile.write(str(line) + '\n')

def cycle_clustering_KMEANS(filepath, K):
    data = readMatrix(filepath, float)
    estimator = KMeans(n_clusters=K)
    estimator.fit(data)
    pred = estimator.labels_
    
    # 【修复：使用 os.path.basename 替换切片】
    filename = os.path.basename(filepath)
    tkpath = 'tmpKmeans_' + filename
    
    with open(tkpath, 'w') as outfile:
        for index, label in enumerate(pred):
            outfile.write(f"{index+1} {label+1}\n")
    return tkpath, pred

def cycle_clustering_SpectralClustering(filepath, K):
    data = readMatrix(filepath, float)
    estimator = SpectralClustering(n_clusters=K, affinity='rbf', gamma=0.1)
    estimator.fit(data)
    pred = estimator.labels_
    
    # 【修复：使用 os.path.basename 替换切片】
    filename = os.path.basename(filepath)
    tkpath = 'tmpSpectral_' + filename
    
    with open(tkpath, 'w') as outfile:
        for index, label in enumerate(pred):
            outfile.write(f"{index+1} {label+1}\n")
    return tkpath, pred

def test_clustering(paths, gtpath, turn=0):
    print(" -> Node clustering evaluation...")
    Gt_raw = readMatrix(gtpath, int)
    maxinCol = Gt_raw.argmax(axis=0)
    mininCol = Gt_raw.argmin(axis=0)
    K = int(Gt_raw[maxinCol[1], 1]) - int(Gt_raw[mininCol[1], 1]) + 1
    
    scores = []
    for path in paths:
        if not os.path.exists(path): continue
        data = readMatrix(path, float)
        # 截取对应长度的 GT 标签
        N_samples = data.shape[0]
        Gt = Gt_raw[:N_samples, :]
        
        _, pred = cycle_clustering_KMEANS(path, K)
        AMI = metrics.adjusted_mutual_info_score(Gt[:, 1], pred)
        ARI = metrics.adjusted_rand_score(Gt[:, 1], pred)
        V = metrics.v_measure_score(Gt[:, 1], pred)
        Si = metrics.silhouette_score(data, pred) if len(set(pred)) > 1 else 0
        scores.append([AMI, ARI, V, Si])
        
    writeMatrixTxt(scores, '00' + str(turn) + 'PyCluster.txt')

def combine_topo(inpath, outpath, add_ratio):
    A = ReadEL(inpath)
    N = np.shape(A)[0]
    with open(inpath, 'r') as f_in, open(outpath, 'w') as f_out:
        orig_content = f_in.read()
        f_out.write(orig_content)
        sample = np.arange(1, N + 1)
        addnum = int(ceil(N * add_ratio))
        np.random.shuffle(sample)
        for i in sample[:addnum]:
            f_out.write(f"{i} {N+i}\n")
        f_in.seek(0)
        for line in f_in:
            a, b = map(int, line.strip().split())
            f_out.write(f"{N+a} {N+b}\n")
    return N

def test_role(paths, N, add):
    rolescore = []
    for path in paths:
        if not os.path.exists(path): continue
        features = readMatrix(path, float)
        # 计算 Cosine 和 L2 识别率
        rawSim = np.matmul(features, features.T)
        mod = np.sqrt(np.diag(rawSim)).reshape(-1, 1) * np.sqrt(np.diag(rawSim))
        mod[mod == 0] = 0.0001
        cosSim = rawSim / mod
        
        # 寻找匹配
        score_cos = 0
        for i in range(N):
            # 排除自身
            row = cosSim[i, :].copy()
            row[i] = -1
            top_match = np.argmax(row)
            if top_match == (i + N): score_cos += 1
        rolescore.append([score_cos / N, 0.0]) # 简化 L2 提升速度
    return rolescore