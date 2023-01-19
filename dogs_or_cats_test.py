#!/usr/bin/env python
# coding: utf-8

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps


# TensorFlowで作成したモデルを導入する
model = tf.keras.models.load_model("src/dogs_or_cats_model.h5")


def sample_predict(img):
	#画像に対して犬/猫の推論をする
	# 画像を160x160の正方形にする
	img = get_square_image(img)
	img = img.resize((160, 160))
	img_array = np.array(img)
	# 推論する
	pred = tf.nn.sigmoid(model.predict(img_array[None, ...]))
	return 1 - pred.numpy()[0][0]  # 1に近いほど猫


def get_result(prediction):
	# 0-1の数値を受け取って表示用のテキストを返す
	if prediction < 0.05:
		result = "確実に犬"
	elif prediction < 0.2:
		result = "ほぼ犬"
	elif prediction < 0.5:
		result = "どちらかといえば犬"
	elif prediction < 0.8:
		result = "どちらかといえば猫"
	elif prediction < 0.95:
		result = "ほぼ猫"
	else:
		result = "確実に猫"
	return result



### 表示部分
st.title("犬猫判別アプリ")

uploaded_file = st.file_uploader("判定したい画像をアップロードしてください")
if uploaded_file is not None:
	# 画像を読み込む
	uploaded_img = Image.open(uploaded_file)
	uploaded_img = ImageOps.exif_transpose(uploaded_img)  # 画像を適切な向きに補正する

	# 犬猫判定
	pred = sample_predict(uploaded_img)

	# 結果表示
	st.info(f"これは**{get_result(pred)}**です！")
	score = np.int(np.round(pred, 2)*20)
	st.text(f"犬 0 |{'-'*score}*{'-'*(19-score)}| 100 猫")
	st.image(uploaded_img, use_column_width=True)
