# sci23

参考  
綾鷹 AI：https://qiita.com/akidon0000/items/8fccd884559521869180  
顔領域の切り取り：https://qiita.com/john-rocky/items/ce008d6dcebf39dd9bf0

### 下準備：

・data フォルダ内の images フォルダの中に train_test_data というフォルダを作り、data.npy という空のファイルを作る

### original の画像データをスクレイピング

$ python get_thumbnail_images.py

### 顔領域の切り抜き

$ python face_cut.py

### 教師データとテストデータを作成

$ python train_test_data_generator.py

### 学習モデルを作成

$ python model_generator.py

### 写真を判定

$ python main.py './data/images/testData/test1.jpg'
