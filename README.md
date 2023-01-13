# sci23

参考：https://qiita.com/akidon0000/items/8fccd884559521869180

データ水増し
$ python image_duplicator.py

教師データとテストデータを作成
$ python model/generator.py

学習モデルを作成
$ python model_generator.py

写真を判定
$ python main.py '../AyatakaAI/data/images/testData/test1.jpg'
