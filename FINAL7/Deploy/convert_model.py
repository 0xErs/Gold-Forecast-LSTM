import tensorflow as tf

model = tf.keras.models.load_model('lstm_gold.keras', compile=False)
model.summary()
model.save('lstm_gold.h5')
print("✅ Selesai! File lstm_gold.h5 sudah dibuat")