# sci23

参考  
綾鷹 AI：https://qiita.com/akidon0000/items/8fccd884559521869180  
顔領域の切り取り：https://qiita.com/john-rocky/items/ce008d6dcebf39dd9bf0  
肌色領域の抽出：https://techtech-sorae.com/pythonopencv%E3%81%A7hsv%E8%89%B2%E7%A9%BA%E9%96%93%E3%82%92%E7%94%A8%E3%81%84%E3%81%A6%E8%82%8C%E8%89%B2%E9%A0%98%E5%9F%9F%E3%82%92%E6%8A%BD%E5%87%BA/  
肌色領域のマスク：https://qiita.com/kotai2003/items/4b3f48db9ef8ae503fa1

### 下準備：

・data フォルダ内の images フォルダの中に train_test_data というフォルダを作り、data.npy という空のファイルを作る

### original の画像データをスクレイピング

$ python get_thumbnail_images.py

### 顔領域の切り抜き

$ python face_cut.py

### 肌色領域の抽出と色の平均値の計算

$ python face_color.py  
$ python get_face_color_avg.py

### 顔のランドマーク検出

$ python face_landmarks.py

### 教師データとテストデータを作成

$ python train_test_data_generator.py

### 学習モデルを作成

$ python model_generator.py

### 写真を判定

$ python main.py './data/images/testData/test1.jpg'
