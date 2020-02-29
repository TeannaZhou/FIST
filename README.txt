cpu运行

1. 测试
在终端Terminal执行
python main.py    # 默认执行test任务，test样本为默认video，在显示画面按q键结束test, 'Negative' 手背朝前, 'Positive' 手心朝前
or
python main.py --test_type=1   # 执行test任务， test样本为默认图片， 检测结果打印在图片上保存在当前目录下
or
python main.py --test_type=1 --test_img="你想要检测的图片的地址"

2. 训练
在终端Terminal运行
(1) 生成训练集及验证集
    cd data
    python generate_dataset.py
(2) 训练
    python main.py --type=train   # 默认训练 90 epochs, batch_size=32 等等，参数可设置，见 main.py
