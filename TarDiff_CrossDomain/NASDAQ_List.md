## 已完成
1. conda activate tardiff
2. cd TarDiff_CrossDomain
3. 从 Kaggle 下载数据集: https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset
4. 解压缩到 TarDiff_CrossDomain/data/raw/NASDAQ
5. python scripts/explore_nasdaq.py --data_path data/raw/NASDAQ 
6. python scripts/preprocess_nasdaq.py --input_path data/raw/NASDAQ --output_path data/processed/nasdaq

---

## 下一步: 训练基础扩散模型

7. 复制配置文件到 TarDiff 目录


8. 切换到 TarDiff 目录并开始训练
```bash
cd ..\TarDiff
python train_main.py --config configs/base/nasdaq_base.yaml --name nasdaq_exp

python train_main.py --base configs/base/nasdaq_base.yaml --name nasdaq_exp --logdir ./outputs
```

9. (可选) 查看训练日志
```bash
tensorboard --logdir logs/nasdaq_exp
```

---

## 输出记录
```
NASDAQ 数据预处理
======================================================================
输入路径: data\raw\NASDAQ
输出路径: data\processed\nasdaq
序列长度: 24
预测窗口: 5 天
采样策略: 非重叠切分 (stride = 24)
最小日期: 2010-01-01
最小数据长度: 1000 天

找到 5884 个股票文件
符合条件的股票: 4668

生成样本 (非重叠切分)...
总样本数: 451960
数据形状: (451960, 5, 24)
正样本比例: 50.28%
平均每只股票: 96.8 个样本

划分数据集...
训练集: 361568 样本 (正样本: 50.26%)
验证集: 45196 样本 (正样本: 50.38%)
测试集: 45196 样本 (正样本: 50.26%)

保存数据...
  保存: data\processed\nasdaq\train_tuple.pkl
  保存: data\processed\nasdaq\val_tuple.pkl
  保存: data\processed\nasdaq\test_tuple.pkl
  保存: data\processed\nasdaq\meta.pkl

======================================================================
  形状: (5, 24) (通道数, 时间步)
  各通道数据:
    Open  : [1.68, 1.70, ... , 1.68, 1.67]
    High  : [1.70, 1.70, ... , 1.71, 1.74]
    Low   : [1.66, 1.68, ... , 1.68, 1.65]
    Close : [1.67, 1.68, ... , 1.71, 1.74]
    Volume: [35100.00, 2000.00, ... , 700.00, 19800.00]

样本 1:
  标签: 1 (上涨)
  形状: (5, 24) (通道数, 时间步)
  各通道数据:
    Open  : [23.93, 24.61, ... , 22.87, 22.91]
    High  : [24.57, 24.63, ... , 22.92, 23.10]
    Low   : [23.93, 23.85, ... , 22.75, 22.78]
    Close : [24.40, 23.93, ... , 22.91, 22.90]
    Volume: [18800.00, 25900.00, ... , 7500.00, 9700.00]

下一步: 配置 TarDiff 并开始训练
```