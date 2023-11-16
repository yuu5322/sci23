# ToDo: MAQUIAがスクレイピングを許可しているかどうか確認する（ソースコードに書いてある場合もある）

import time
import requests
import re
import uuid
from bs4 import BeautifulSoup

# 200ページ分（1ページは約12記事）のサムネイル画像を取得
# 多分1時間弱くらいかかった
for page in range(1,201):
   r = requests.get("https://maquia.hpplus.jp/makeup/news/?page="+str(page))
   time.sleep(1)
   r.encoding = r.apparent_encoding
   response = r.text

   soup = BeautifulSoup(response,'html.parser')
   article_list = soup.find('ul', class_='article-list')
   articles = article_list.findAll('div', class_='article-card-media')

   for article in articles:
      imgs = article.find_all('img')
      for img in imgs:
         r = requests.get(img['src'])
         time.sleep(1)
         with open(str('./images/original/')+str(uuid.uuid4())+str('.jpeg'),'wb') as file:
               file.write(r.content)

