import streamlit as st 

# tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import  Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow.keras.utils as ku

from keras.models import load_model
import tensorflow as tf

import numpy as np
model = tf.keras.models.load_model('quotes_2.h5')

def main():
	st.title("Love Poetry Generated AI ")

	st.sidebar.title("Himanshu Tripathi")
	st.sidebar.header("AI-Generated Poetry")
	st.sidebar.subheader('''
		Have you ever try to write "Poetry". But don't know what to write and have you try to impress the friends of your loved ones.
		But you're not able to do it write one... 
		Don't Worry now "A.I" will do this for you.''')
	max_sequence_len = 74
	# 

	tokenizer = Tokenizer()
	data = open('text_file.txt',encoding='utf-8').read()

	corpus = data.lower().split('\n')

	tokenizer.fit_on_texts(corpus)
	# 
	# st.TextBox("Write message")

	

	# seed_text = "the word"
	# next_words = 30
	text = st.text_input("Enter your text to make Poetry")
	seed_text = text 
	length = st.slider(label = 'Poetry length',min_value=20, max_value=50)
	next_words = length
	if st.button("Generate Poerty"):
		with st.spinner("Just wait a second.. Making Something good for you... "):
			for _ in range(next_words):
				token_list = tokenizer.texts_to_sequences([seed_text])[0]
				token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
				predicted = model.predict_classes(token_list, verbose=0)
				# st.write(predicted)
				output_word = ""
				for word, index in tokenizer.word_index.items():
					if index == predicted:
						output_word = word
						# st.write(output_word)
						break
				seed_text += " " + output_word

			# print(seed_text)
		st.success(seed_text)



	hide_footer_style = """<style>
    .reportview-container .main footer {visibility : hidden;}
      
    """
	st.markdown(hide_footer_style, unsafe_allow_html=True)



if __name__ == '__main__':
	main()