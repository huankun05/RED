import numpy as np
import diameter

def ReadEL(filein):
    fin = open(filein, 'r')
    N_max = 0
    f_lines = fin.readlines()
    for line in f_lines:
        line = line.strip()
        if not line: continue
        vec = line.split()
        u, v = int(vec[0]), int(vec[1])
        if u > N_max: N_max = u
        if v > N_max: N_max = v
    fin.close()
    A = np.zeros((N_max, N_max), dtype=float)
    fin = open(filein, 'r')
    for line in fin:
        line = line.strip()
        if not line: continue
        vec = line.split()
        a, b = int(vec[0]) - 1, int(vec[1]) - 1
        A[a][b] = 1.0; A[b][a] = 1.0
    fin.close()
    return A

def Szegedy_Core_Step(psi, W, A):
    """Szegedy 演化步: U = S(2*Pi - I)"""
    # 投影步
    coeffs = np.sum(psi * W, axis=1, keepdims=True)
    # 反射步 + 交换步 (S 为转置)
    new_psi = (2.0 * coeffs * W - psi).T
    return new_psi

def RED_Optimized(inpath, outpath, dim, d_global, tq=3):
    """
    V5 优化版：Szegedy 算子 + 节点特异性干涉对比度指纹
    """
    A = ReadEL(inpath)
    N = A.shape[0]
    deg = np.sum(A, axis=1).reshape(-1, 1)
    deg[deg == 0] = 1.0
    W = np.sqrt(A / deg) # Szegedy 投影基

    # --- 1. 全局分支：节点量子指纹 ---
    # 初始态：全网均匀叠加
    psi_g = W.copy().astype(complex) / np.sqrt(N)
    steps_g = int(d_global / 2)
    if steps_g < 1: steps_g = 1
    
    global_features = np.zeros((N, steps_g * 2))
    for t in range(steps_g):
        psi_g = Szegedy_Core_Step(psi_g, W, A)
        
        # 节点特异性概率 (不再是全局熵)
        prob = np.sum(np.abs(psi_g)**2, axis=1).flatten()
        # 节点特异性相干干涉项 (V3 的核心增益来源)
        interference = np.abs(np.sum(psi_g, axis=1))**2 - prob
        
        global_features[:, t] = prob
        global_features[:, steps_g + t] = interference.flatten()
    
    # 全局特征归一化
    g_min, g_max = global_features.min(), global_features.max()
    if g_max > g_min:
        global_features = (global_features - g_min) / (g_max - g_min)

    # --- 2. 局部分支：基于 Szegedy 的累积紧密度 ---
    try: dia = diameter.diameter(inpath)
    except: dia = 5
    
    proximity_matrix = np.zeros((N, N))
    for i in range(N):
        # 针对每个节点 i 的偏置初始态
        psi_l = np.zeros((N, N), dtype=complex)
        psi_l[i, :] = W[i, :]
        
        acc_prob = np.zeros(N)
        for _ in range(dia):
            psi_l = Szegedy_Core_Step(psi_l, W, A)
            acc_prob += np.sum(np.abs(psi_l)**2, axis=1).flatten()
        proximity_matrix[i, :] = acc_prob

    # --- 3. 融合与导出 ---
    d_local = dim - global_features.shape[1]
    local_feature = -np.sort(-proximity_matrix)[:, :max(0, d_local)]
    # 局部特征归一化
    l_min, l_max = local_feature.min(), local_feature.max()
    if l_max > l_min:
        local_feature = (local_feature - l_min) / (l_max - l_min)

    final_emb = np.hstack((global_features, local_feature))
    
    import test
    test.writeMatrixTxt(final_emb.tolist(), outpath + '.txt')
    print(f"V5 Szegedy-Interference RED finished | Dim: {final_emb.shape[1]}")
    return [outpath + '.txt']