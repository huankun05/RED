import numpy as np
from scipy.sparse import csr_matrix
import time

def get_sparse_W(inpath):
    # 快速读入并构造稀疏环境
    edges = np.loadtxt(inpath, dtype=int)
    u, v = edges[:, 0] - 1, edges[:, 1] - 1
    n = max(u.max(), v.max()) + 1
    adj = csr_matrix((np.ones(len(u)), (u, v)), shape=(n, n))
    adj = adj + adj.T
    adj.data = np.ones_like(adj.data)
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv = 1.0 / np.where(deg == 0, 1.0, deg)
    W_dense = adj.multiply(np.sqrt(deg_inv)[:, np.newaxis]).toarray()
    return W_dense, n, deg, adj

def szegedy_step_fast(psi, W_dense):
    coeffs = np.einsum('ij,ij->i', psi, W_dense)[:, np.newaxis]
    return (2.0 * coeffs * W_dense - psi).T

def RED_Optimized(inpath, outpath, dim, d_global, tq=3, gamma=0.75):
    start_time = time.time()
    W_dense, N, deg, adj = get_sparse_W(inpath)
    
    # --- 1. 全局快照分支 (Snapshots + Log-Transform) ---
    snapshots = [2, 4, 6] 
    psi = (W_dense / np.sqrt(N)).astype(np.complex64)
    
    raw_features = []
    # 基础物理特征：Log-Degree
    raw_features.append(np.log1p(deg).reshape(-1, 1))
    
    for t in range(1, max(snapshots) + 1):
        psi = szegedy_step_fast(psi, W_dense)
        if t in snapshots:
            prob = np.sum(np.abs(psi)**2, axis=1)
            inter = np.abs(np.sum(psi, axis=1))**2 - prob
            
            # 引入 Log 变换和相干比率，增强 ARI
            raw_features.append(np.log1p(prob).reshape(-1, 1))
            # 相干比率特征
            coherence_ratio = inter / (prob + 1e-9)
            raw_features.append(coherence_ratio.reshape(-1, 1))
            
    global_feat = np.hstack(raw_features)
    
    # --- 2. 邻域上下文聚合 (Contextual Smoothing - 提升 ARI 的秘诀) ---
    # 利用稀疏矩阵进行一次快速的邻域特征平均，增强特征的平滑度
    # D^-1 * A * Feature
    deg_norm = 1.0 / (deg + 1e-9)
    # 将稀疏度数矩阵与特征相乘
    smoothed_feat = adj.multiply(deg_norm[:, np.newaxis]).dot(global_feat)
    
    # 结合原始特征与平滑特征
    combined_global = np.hstack((global_feat, smoothed_feat))
    combined_global = (combined_global - combined_global.mean(axis=0)) / (combined_global.std(axis=0) + 1e-9)
    
    # --- 3. 局部分支：带拓扑偏置的扩散 ---
    psi_l = W_dense.copy().astype(np.complex64)
    proximity = np.zeros((N, N), dtype=np.float32)
    curr_gamma = 1.0
    for _ in range(4):
        psi_l = szegedy_step_fast(psi_l, W_dense)
        proximity += np.abs(psi_l)**2 * curr_gamma
        curr_gamma *= gamma
    
    # --- 4. 融合与降维 ---
    d_local = dim - combined_global.shape[1]
    if d_local > 0:
        local_feat = -np.sort(-proximity, axis=1)[:, :d_local]
        local_feat = (local_feat - local_feat.mean(axis=0)) / (local_feat.std(axis=0) + 1e-9)
        emb = np.hstack((combined_global, local_feat))
    else:
        # 如果全局特征已经足够，利用主成分思想取前 dim 维
        emb = combined_global[:, :dim]

    # 最终映射
    emb = (emb - emb.min()) / (emb.max() - emb.min() + 1e-9)
    
    import test
    test.writeMatrixTxt(emb.tolist(), outpath + '.txt')
    
    print(f"V5.5 Physics-Adaptive RED | Time: {time.time()-start_time:.4f}s")
    return [outpath + '.txt']