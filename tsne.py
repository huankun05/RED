import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets

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

# def HighVisualize(path,gtpath):
#     features = readMatrix(path, float)
#     gtmatrix = readMatrix(gtpath, int)
#     gtvec = np.reshape(gtmatrix[:,1],(np.shape(features)[0],))
#     X, y = features, gtvec
#     n_samples, n_features = X.shape

#     tsne = manifold.TSNE(n_components=2, early_exaggeration	= 20, learning_rate=10, init='pca')
#     X_tsne = tsne.fit_transform(X)
#     x_min, x_max = X_tsne.min(0), X_tsne.max(0)
#     xmargin = (x_max-x_min)*0.25
#     #X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化

#     plt.figure(figsize=(8, 5))

#     plt.axis([x_min[0]-xmargin[0],x_max[0]+xmargin[0],x_min[1]-xmargin[1],x_max[1]+xmargin[1]])
#     plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y, cmap='brg',edgecolors='black', s=100) #'gist_rainbow' cmap='brg'
#     #for i in range(X_tsne.shape[0]):
#         #plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), fontdict={'weight': 'bold', 'size': 9})
#     #    plt.text(X_tsne[i, 0], X_tsne[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), fontdict={'weight': 'bold', 'size': 9})
#     #plt.xticks([])
#     #plt.yticks([])
#     plt.title(path[7:-4], fontsize=18, fontweight='roman')
#     plt.xlabel('t-SNE 1', fontsize=14, fontweight='light')
#     plt.ylabel('t-SNE 2', fontsize=14, fontweight='light')

#     ax = plt.gca()
#     ax.spines['top'].set_linewidth(1.5)
#     ax.spines['bottom'].set_linewidth(1.5)
#     ax.spines['right'].set_linewidth(1.5)
#     ax.spines['left'].set_linewidth(1.5)


#     plt.savefig(path[:-3] + 'png')
#     #plt.show()
#     return 0

def HighVisualize(path, gtpath):
    # 1. 读取嵌入特征特征
    features = readMatrix(path, float)
    
    # 2. 读取原始 Ground Truth 标签
    gtmatrix_raw = readMatrix(gtpath, int)
    
    # --- 修复逻辑：确保标签行数与特征行数严格一致 ---
    # 获取当前特征矩阵的节点数量 (n_samples)
    n_samples = features.shape[0]
    # 仅截取前 n_samples 行标签，防止 154 对 77 的不匹配报错
    gtmatrix = gtmatrix_raw[:n_samples, :]
    # --------------------------------------------
    
    # 提取标签列（通常是第二列）并重塑形状
    gtvec = np.reshape(gtmatrix[:, 1], (n_samples,))
    X, y = features, gtvec
    
    print(f"Generating t-SNE for {path}... (Samples: {n_samples})")

    # 3. 初始化 t-SNE 模型
    # n_components=2: 降维到2维用于绘图
    # init='pca': 使用 PCA 初始化可以增加结果的稳定性
    tsne_model = manifold.TSNE(
        n_components=2, 
        early_exaggeration=20, 
        learning_rate=10, 
        init='pca', 
        random_state=42 # 固定随机种子使结果可复现
    )
    
    # 执行降维
    X_tsne = tsne_model.fit_transform(X)
    
    # 4. 设置绘图边界和边距
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    xmargin = (x_max - x_min) * 0.25

    # 5. 开始绘图
    plt.figure(figsize=(8, 5))
    plt.axis([
        x_min[0] - xmargin[0], x_max[0] + xmargin[0], 
        x_min[1] - xmargin[1], x_max[1] + xmargin[1]
    ])
    
    # 绘制散点图
    # c=y: 根据角色标签着色
    # cmap='brg': 颜色映射方案
    # edgecolors='black': 增加黑色轮廓使点更清晰
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='brg', edgecolors='black', s=100)
    
    # 设置标题和轴标签
    # path[7:-4] 截取文件名作为标题，例如 "orig" 或 "optimized"
    plt.title(path[7:-4], fontsize=18, fontweight='roman')
    plt.xlabel('t-SNE 1', fontsize=14, fontweight='light')
    plt.ylabel('t-SNE 2', fontsize=14, fontweight='light')

    # 美化边框线宽
    ax = plt.gca()
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)

    # 保存图片
    # 将 .txt 后缀替换为 .png
    output_png = path[:-3] + 'png'
    plt.savefig(output_png)
    # plt.show() # 如果需要实时查看可以取消注释
    
    print(f"Image saved to {output_png}")
    return 0

def test_highvisualize(paths):
    for path in paths:
        HighVisualize(path, 'gt.txt')

if __name__ == "__main__":
    HighVisualize(path='matrix_GraphWave2.txt', gtpath='gt.txt')