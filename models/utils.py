from models.neural_networks import MLP
import numpy as np
# DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def create_model(config, device='cpu'):
    """Creates and returns a neural network model."""
    input_dim = config["input_dim"]
    output_dim = config["output_dim"]
    hidden_dim = config["hidden_dim"]
    dropout = config["dropout"]
    model = MLP(input_dim, hidden_dim, output_dim, dropout=dropout)
    return model.to(device)


def path_clean(path, target):
    """
    清理路径，去除重复点和距离过近的点
    """
    # # 找到离终点最近的点
    dists_to_target = np.linalg.norm(path[:, :2] - target, axis=1)
    min_index = np.argmin(dists_to_target)
    path = path[:min_index + 1]
    # # print(path)

    i = 1
    while i < len(path):
        min_index = i-1
        min_dist = np.linalg.norm(path[i, :2] - path[min_index, :2])
        for j in range(0, i-1):
            dist = np.linalg.norm(path[j, :2] - path[i, :2])
            if dist < min_dist:
                min_dist = dist
                min_index = j
        if min_index != i-1:
            # 删掉从 min_index+1 到 i-1 的点
            path = np.delete(path, np.s_[min_index+1:i], axis=0)
            i = min_index + 1
        else:
            i += 1
    # return path
    cleaned_path = [path[0]]
    i=1
    while i < len(path)-1:
        next_p = None
        next_index = i
        for j in range(i, len(path)-1):
            dist = np.linalg.norm(path[j, :2] - cleaned_path[-1][:2])
            if dist >= 0.5:  # 保留距离大于阈值的点
                next_p = path[j]
                next_index = j
                break
        if next_p is not None:
            cleaned_path.append(next_p)
        i = next_index if next_index > i else i + 1  # 防止死循环
    cleaned_path.append(path[-1])  # 确保终点被添加
    return np.array(cleaned_path)