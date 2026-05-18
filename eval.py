import speck as sp
import numpy as np

import argparse

# Keras 可能有两种安装形式：独立的 `keras` 包，或者 TensorFlow 自带的
# `tensorflow.keras`。这里先尝试原论文代码使用的独立 Keras，如果不可用，
# 再兼容新版 TensorFlow 环境。
try:
    from keras.models import model_from_json
except ImportError:
    from tensorflow.keras.models import model_from_json

# 读取 5/6/7/8 轮神经区分器共用的网络结构。
# 网络结构保存在 JSON 文件中，而不同轮数的训练参数保存在后面的 .h5 权重文件中。
json_file = open('single_block_resnet.json','r');
json_model = json_file.read();

# 根据同一个 JSON 网络结构创建四个模型。
# 它们一开始结构相同，加载不同轮数的权重后才分别变成 N5/N6/N7/N8 区分器。
net5 = model_from_json(json_model);
net6 = model_from_json(json_model);
net7 = model_from_json(json_model);
net8 = model_from_json(json_model);

# 加载仓库提供的预训练权重。
# 这些权重分别对应 Speck32/64 的 5、6、7、8 轮神经区分器。
net5.load_weights('net5_small.h5');
net6.load_weights('net6_small.h5');
net7.load_weights('net7_small.h5');
net8.load_weights('net8_small.h5');

def evaluate(net,X,Y):
    # 运行神经区分器进行预测。
    # 每个输出分数可以理解为：模型认为该密文对是“真实 Speck 加密对”的置信度。
    Z = net.predict(X,batch_size=10000).flatten();

    # 把连续分数转换成二分类结果。
    # 分数大于 0.5 判为真实对，否则判为随机对。
    Zbin = (Z > 0.5);

    # 均方误差 MSE：衡量模型原始输出分数和真实标签 0/1 的距离。
    diff = Y - Z; mse = np.mean(diff*diff);

    # 统计总样本数，以及随机类和真实类各自的样本数。
    n = len(Z); n0 = np.sum(Y==0); n1 = np.sum(Y==1);

    # Accuracy：所有样本中预测正确的比例。
    acc = np.sum(Zbin == Y) / n;

    # TPR（真正率）：真实 Speck 对中，有多少被模型正确判为真实。
    tpr = np.sum(Zbin[Y==1]) / n1;

    # TNR（真负率）：随机对中，有多少被模型正确判为随机。
    tnr = np.sum(Zbin[Y==0] == 0) / n0;

    # 真实 Speck 对预测分数的中位数。
    mreal = np.median(Z[Y==1]);

    # 随机对中，有多少比例的分数高于“真实对分数中位数”。
    # 这个指标反映真实对和随机对的分数分布重叠程度，越低越好。
    high_random = np.sum(Z[Y==0] > mreal) / n0;

    print("Accuracy: ", acc, "TPR: ", tpr, "TNR: ", tnr, "MSE:", mse);
    print("Percentage of random pairs with score higher than median of real pairs:", 100*high_random);

def run_eval(samples_per_round=10**6):
    # 生成普通 real-vs-random 测试数据，轮数分别为 5 到 8 轮。
    # 这对应论文主区分器表格中的实验设置：
    # 标签 1 表示真实 Speck 加密对，标签 0 表示随机对。
    X5,Y5 = sp.make_train_data(samples_per_round,5);
    X6,Y6 = sp.make_train_data(samples_per_round,6);
    X7,Y7 = sp.make_train_data(samples_per_round,7);
    X8,Y8 = sp.make_train_data(samples_per_round,8);

    # 生成 real differences（真实差分）测试数据。
    # 这是更难的实验设置：两类样本都来自真实 Speck 加密，
    # 但只有一类使用区分器期望的目标输入差分。
    X5r, Y5r = sp.real_differences_data(samples_per_round,5);
    X6r, Y6r = sp.real_differences_data(samples_per_round,6);
    X7r, Y7r = sp.real_differences_data(samples_per_round,7);
    X8r, Y8r = sp.real_differences_data(samples_per_round,8);

    # 在普通 real-vs-random 设置下评估四个预训练网络。
    print('Testing neural distinguishers against 5 to 8 blocks in the ordinary real vs random setting');
    print('5 rounds:');
    evaluate(net5, X5, Y5);
    print('6 rounds:');
    evaluate(net6, X6, Y6);
    print('7 rounds:');
    evaluate(net7, X7, Y7);
    print('8 rounds:');
    evaluate(net8, X8, Y8);

    # 在 real differences 设置下评估同一批网络。
    print('\nTesting real differences setting now.');
    print('5 rounds:');
    evaluate(net5, X5r, Y5r);
    print('6 rounds:');
    evaluate(net6, X6r, Y6r);
    print('7 rounds:');
    evaluate(net7, X7r, Y7r);
    print('8 rounds:');
    evaluate(net8, X8r, Y8r);

def parse_args():
    # 解析命令行参数，用来控制测试样本数量。
    # 论文中的神经区分器结果使用的是 10^6 个样本。
    parser = argparse.ArgumentParser(description='Evaluate pre-trained Speck distinguishers (5-8 rounds).')
    parser.add_argument('--samples', type=int, default=10**6, help='Samples per round for each setting (default: 1e6).')
    return parser.parse_args()

if __name__ == '__main__':
    # 脚本入口：读取命令行参数，然后运行两类评估实验。
    args = parse_args()
    run_eval(samples_per_round=args.samples)
