# imports
import cv2
import tensorflow as tf
CATEGORIES = [
    "Arroz Dia 5kg",
    "Arroz Ecco 5Kg",
    "Arroz pedreti 5Kg",
    "Café 3 coração Extra forte 500g",
    "Café 3 coração Tradicional 500g",
    "café Dia 1Kg",
    "Feijao Carioca camil 1kg",
    "Feijão Carioca Kicaldo 1Kg",
    "Feijao Carioca Pedretti 2Kg",
    "Leite Batavo 1L",
    "Leite Dia 1L",
    "Leite Jussara 1L",
    "Macarrao espaguete Brasilar 500g",
    "Macarrão parafuso Brasilar 500g",
    "Sache molho de tomate Dia 340g",
    "Sache molho de tomate Tarantela 340g"  ,
    "Arroz Dia>>>>",
    "Arroz Ecco>>>> 5Kg",
    "Arroz pedreti>>>> 5Kg",
    "Café 3 coração Extra forte>>>>>>> 500g",
    "Café 3 coração Tradicional>>>>>>> 500g",
    "café Dia>>>>>>> 1Kg",
    "Feijao Carioca camil>>>>>>>> 1kg",
    "Feijão Carioca Kicaldo>>>>>>>> 1Kg",
    "Feijao Carioca Pedretti>>>>>>>> 2Kg",
    "Leite Batavo>>>>>>>> 1L",
    "Leite Dia>>>>>>>> 1L",
    "Leite Jussara>>>>>>>> 1L",
    "Macarrao espaguete Brasilar>>>>>>>> 500g",
    "Macarrão parafuso Brasila>>>>>>>>r 500g",
    "Sache molho de tomate Dia>>>>>>>> 340g",
    "Sache molho de tomate Tarantela>>>>>>>> 340g"
    ]

# prepare function
def prepare(filepath):
    IMG_SIZE_H = 216
    IMG_SIZE_W = 438
    img_array = cv2.imread(filepath)
    # img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE_W, IMG_SIZE_H))
    return new_array.reshape(-1, IMG_SIZE_W, IMG_SIZE_H, 3)

# loading model
model = tf.keras.models.load_model("C:\ImagensIC\Teste_sameSize-16class\modelTHL1.h5")

# making predictions
prediction = model.predict([prepare('C:\ImagensIC\Teste_sameSize-16class\Train5911.jpg')])
print('First image is ' + CATEGORIES[int(prediction[0])])
print(CATEGORIES[int(prediction[0][0])])