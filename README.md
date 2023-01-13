# sci23

参考：https://qiita.com/akidon0000/items/8fccd884559521869180

データ水増し
$ cd /../img_dumplicator
$ python main.py

教師データとテストデータを作成
$ cd /../train_test_data_generator
$ python main.py

学習モデルを作成
$ cd /../model_generator
$ python main.py

写真を判定
cd /../ai
$ python main.py '../AyatakaAI/data/images/testData/test1.jpg'
