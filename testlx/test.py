# %%
import numpy as np

def calculate_neuron_scores(outputs):
    """
    计算神经元评分 (公式 3)
    :param outputs: 形状为 (样本数, 神经元数) 的二维数组
    """
    # 1. 计算每个神经元输出绝对值的期望 (分子部分)
    # np.abs 取绝对值, axis=0 表示按列(神经元)取平均
    expected_abs_output = np.mean(np.abs(outputs), axis=0)
    
    # 2. 计算所有神经元期望的平均值 (分母部分)
    layer_mean_expectation = np.mean(expected_abs_output)
    

    # 3. 计算得分
    scores = expected_abs_output / layer_mean_expectation
    
    return expected_abs_output, layer_mean_expectation, scores

# --- 使用你提供的数值进行测试 ---
# 每一行代表一个输入样本在三个神经元上的输出
# x_a -> (0.1, 9, 6)
# x_b -> (0.2, 7, 1)
# x_c -> (0, 13, 9)
data = np.array([
    [0.1, 9, 6],
    [-0.9, 9, 1],
    [0,   9, 9]
])

exp_vals, layer_avg, final_scores = calculate_neuron_scores(data)

print("--- 神经元评分计算结果 ---")
for i, score in enumerate(final_scores):
    print(f"神经元 {i+1}:")
    print(f"  期望绝对输出: {exp_vals[i]:.4f}")
    print(f"  最终评分 (s_i): {score:.4f}")
    if score <= 0.1:
        print("  状态: 【休眠 (Dormant)】⚠️")
    else:
        print("  状态: 活跃")

print(f"\n全层平均期望值: {layer_avg:.4f}")