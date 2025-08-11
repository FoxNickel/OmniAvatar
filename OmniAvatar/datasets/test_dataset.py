import time

def test():
    print("Testing WanVideoDataset...")
    from datasets import WanVideoDataset
    from torch.utils.data import DataLoader

    dataset_base_path = "/home/huanglingyu/data/vgg/datasets/Koala-36M-v1"

    start_time = time.time()  # 计时开始

    # 实例化数据集
    dataset = WanVideoDataset(dataset_base_path)

    # 打印数据集长度
    print("Dataset length:", len(dataset))

    # 随机取几条数据看看
    for i in range(3):
        data = dataset[i]
        print(f"Sample {i}:")
        for k, v in data.items():
            print(f"  {k}: {v}")
        print("-" * 30)

    # 用 DataLoader 测试 batch 读取
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for batch in dataloader:
        print("Batch keys:", batch.keys() if isinstance(batch, dict) else type(batch))
        print("Batch sample:", batch)
        break  # 只看第一个 batch

    end_time = time.time()  # 计时结束
    print(f"Total time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    test()