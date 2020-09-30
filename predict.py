from keras.models import load_model
import pickle
from keras.preprocessing import sequence

model = load_model('bilstm.h5')
with open('tokenizer_bilstm.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

text = [" User review given as input for predicting the rating"]
txts = tokenizer.texts_to_sequences(text)
txts = sequence.pad_sequences(txts, maxlen= 150 )
preds = model.predict(txts)
print(preds)
