# sci23

参考：https://qiita.com/akidon0000/items/8fccd884559521869180

下準備：
・images フォルダの中に extended という空のフォルダを用意する

データ水増し
$ python img_duplicator.py

教師データとテストデータを作成
$ python model/generator.py

学習モデルを作成
$ python model_generator.py

写真を判定
$ python main.py '../AyatakaAI/data/images/testData/test1.jpg'
