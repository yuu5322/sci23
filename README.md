# sci23

参考  
綾鷹 AI：https://qiita.com/akidon0000/items/8fccd884559521869180  
顔領域の切り取り：https://qiita.com/john-rocky/items/ce008d6dcebf39dd9bf0  
肌色領域の抽出：https://techtech-sorae.com/pythonopencv%E3%81%A7hsv%E8%89%B2%E7%A9%BA%E9%96%93%E3%82%92%E7%94%A8%E3%81%84%E3%81%A6%E8%82%8C%E8%89%B2%E9%A0%98%E5%9F%9F%E3%82%92%E6%8A%BD%E5%87%BA/  
肌色領域のマスク：https://qiita.com/kotai2003/items/4b3f48db9ef8ae503fa1  
肌色領域の平均値（RGB）を計算する：https://qiita.com/ZESSU/items/40b8bb2cd179371df6ac　
顔のランドマーク検出：https://qiita.com/mimitaro/items/bbc58051104eafc1eb38
唇領域の平均値を計算：https://teratail.com/questions/145136, https://imagingsolution.net/program/python/numpy/rgb2bgr_conversion/  
モデル生成：https://qiita.com/tomo_20180402/items/e8c55bdca648f4877188

`model_generator.py`参考  
`keras.models.Sequential`について：https://qiita.com/note-tech/items/bfbd2d63addd490d58ed

※実験し直す場合は、data/images 内の画像を全て削除してから、data_models 内の前回の実験結果のプロットを保存しておくこと

### original の画像データをスクレイピング

$ python get_thumbnail_images.py

### 顔領域の切り抜き

$ python face_cut.py

### 肌色領域の抽出と色の平均値の計算

$ python face_color.py  
$ python get_face_color_avg.py

### 顔のランドマーク検出

$ python face_landmarks.py

### 唇領域の色の平均値の計算

$ python get_lip_color_avg.py

### 教師データとテストデータを作成

綾鷹 AI では`$ python img_duplicator.py`で画像の回転、拡大縮小によるデータの水増しを行なっているが、今回はやらない  
$ python train_test_data_generator.py

### 学習モデルを作成

$ python model_generator.py

### 写真を判定

$ python main.py './data/images/testData/test1.jpg'
