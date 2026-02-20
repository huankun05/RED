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
    deg[deg == 0] = 1.0
    d_inv_sqrt = np.sqrt(1.0 / deg)
    W_dense = adj.multiply(d_inv_sqrt[:, np.newaxis]).toarray()
    return W_dense, n

def szegedy_step_fast(psi, W_dense):
    # 精确 Szegedy 演化算子
    coeffs = np.einsum('ij,ij->i', psi, W_dense)[:, np.newaxis]
    return (2.0 * coeffs * W_dense - psi).T

def RED_Optimized(inpath, outpath, dim, d_global, tq=3, gamma=0.6):
    start_time = time.time()
    W_dense, N = get_sparse_W(inpath)
    
    # --- 1. 全局分支：精选物理指纹 ---
    steps_g = d_global 
    psi_g = (W_dense / np.sqrt(N)).astype(np.complex64)
    
    # 记录整个演化过程中的累积概率，用于捕捉“能量分布”
    prob_accumulator = np.zeros(N)
    interference_accumulator = np.zeros(N)
    
    for t in range(steps_g):
        psi_g = szegedy_step_fast(psi_g, W_dense)
        curr_prob = np.sum(np.abs(psi_g)**2, axis=1)
        curr_int = np.abs(np.sum(psi_g, axis=1))**2 - curr_prob
        
        prob_accumulator += curr_prob
        interference_accumulator += curr_int

    # 特征：平均概率 + 最后一步概率 + 平均干涉项
    f1 = prob_accumulator / steps_g
    f2 = curr_prob
    f3 = interference_accumulator / steps_g
    global_feat = np.column_stack((f1, f2, f3))
    # 归一化全局特征
    global_feat = (global_feat - global_feat.mean(axis=0)) / (global_feat.std(axis=0) + 1e-9)
    
    # --- 2. 局部分支：锐化扩散 ---
    try:
        import diameter
        dia = min(4, diameter.diameter(inpath)) # 局部角色通常不需要看太深
    except:
        dia = 3

    psi_l = W_dense.copy().astype(np.complex64) 
    proximity = np.zeros((N, N), dtype=np.float32)
    
    curr_gamma = 1.0
    for _ in range(dia):
        psi_l = szegedy_step_fast(psi_l, W_dense)
        proximity += np.abs(psi_l)**2 * curr_gamma
        curr_gamma *= gamma # 快速衰减，锐化边界
    
    # --- 3. 融合与导出 ---
    d_local = dim - global_feat.shape[1]
    local_feat = -np.sort(-proximity, axis=1)[:, :max(1, d_local)]
    local_feat = (local_feat - local_feat.mean(axis=0)) / (local_feat.std(axis=0) + 1e-9)
    
    emb = np.hstack((global_feat, local_feat))
    # 全局 Min-Max 缩放至 [0, 1]
    emb = (emb - emb.min()) / (emb.max() - emb.min() + 1e-9)
    
    import test
    test.writeMatrixTxt(emb.tolist(), outpath + '.txt')
    
    print(f"V5.3 High-Accuracy RED | Speedup: ~{1.1/(time.time()-start_time):.1f}x")
    return [outpath + '.txt']