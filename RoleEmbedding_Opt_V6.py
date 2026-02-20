import numpy as np
from scipy.sparse import csr_matrix
import time

def get_sparse_W(inpath):
    edges = np.loadtxt(inpath, dtype=int)
    u, v = edges[:, 0] - 1, edges[:, 1] - 1
    n = max(u.max(), v.max()) + 1
    adj = csr_matrix((np.ones(len(u)), (u, v)), shape=(n, n))
    adj = adj + adj.T
    adj.data = np.ones_like(adj.data)
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv = 1.0 / np.where(deg == 0, 1.0, deg)
    W_dense = adj.multiply(np.sqrt(deg_inv)[:, np.newaxis]).toarray()
    return W_dense, n, deg

def szegedy_step_fast(psi, W_dense):
    coeffs = np.einsum('ij,ij->i', psi, W_dense)[:, np.newaxis]
    return (2.0 * coeffs * W_dense - psi).T

def RED_Optimized(inpath, outpath, dim, d_global, tq=3):
    start_time = time.time()
    W_dense, N, deg = get_sparse_W(inpath)
    
    # --- 1. 全局分支：尺度无关快照 ---
    snapshots = [2, 4, 6, 8] # 增加采样深度以应对更大Motif
    psi = (W_dense / np.sqrt(N)).astype(np.complex64)
    
    prob_snapshots = []
    inter_snapshots = []
    
    for t in range(1, max(snapshots) + 1):
        psi = szegedy_step_fast(psi, W_dense)
        if t in snapshots:
            p = np.sum(np.abs(psi)**2, axis=1)
            # 引入密度补偿：P * deg 能够消除尺寸抖动的影响
            p_norm = p * (deg + 1)
            i = np.abs(np.sum(psi, axis=1))**2 - p
            prob_snapshots.append(p_norm)
            inter_snapshots.append(i)
            
    # 特征 A: 归一化后的概率快照序列
    feat_p = np.column_stack(prob_snapshots)
    # 特征 B: 概率随时间演化的波动率 (区分 Chain vs Cycle 的核心)
    feat_v = np.var(prob_snapshots, axis=0).reshape(-1, 1)
    # 特征 C: 结构干涉指纹
    feat_i = np.column_stack(inter_snapshots)
    
    global_feat = np.hstack((feat_p, feat_v, feat_i))
    # 标准化
    global_feat = (global_feat - global_feat.mean(axis=0)) / (global_feat.std(axis=0) + 1e-9)
    
    # --- 2. 局部分支：修正后的扩散 ---
    psi_l = W_dense.copy().astype(np.complex64)
    proximity = np.zeros((N, N), dtype=np.float32)
    gamma = 0.8
    for _ in range(4):
        psi_l = szegedy_step_fast(psi_l, W_dense)
        proximity += np.abs(psi_l)**2 * gamma
        gamma *= 0.8
        
    # --- 3. 融合 ---
    d_local = dim - global_feat.shape[1]
    if d_local > 0:
        local_feat = -np.sort(-proximity, axis=1)[:, :d_local]
        local_feat = (local_feat - local_feat.mean(axis=0)) / (local_feat.std(axis=0) + 1e-9)
        emb = np.hstack((global_feat, local_feat))
    else:
        # 如果全局特征已经占满维度，使用 SVD 降维保留最强分量
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=dim)
        emb = svd.fit_transform(global_feat)

    # 最终映射
    emb = (emb - emb.min()) / (emb.max() - emb.min() + 1e-9)
    
    import test
    test.writeMatrixTxt(emb.tolist(), outpath + '.txt')
    
    print(f"V6.2 Multi-Scale RED | Nodes: {N} | Speed: {time.time()-start_time:.4f}s")
    return [outpath + '.txt']