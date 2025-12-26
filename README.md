## 概要
**CLIP**を用いてデータセットから任意の物体のみを取り出す
## 環境構築
### CLIP
```
conda create -n clip python=3.8 -y
conda activate clip
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

## 使い方
画像データセット内から車の画像を取り出したい場合の例
```bash
python ./clip_images.py --input_dir path/to/dataset/ --text "a photo of the exterior of a car" --dest_dir "./car_images" --threshold 0.9
```

車のデータセット内から任意の車の画像を取り出したい場合の例
```bash
python ./clip_images.py --input_dir path/to/dataset/ --text "a photo of the exterior of a red car" --dest_dir "./car_images" --threshold 0.9
```

CLIPを用いる際はネガティブプロンプトを指定した方が精度があがる \
以下はtextに`a red car`と指定した場合の例
```python
texts = [target_text, "a blue car", "a orange car", "a white car", "a black car"]
```