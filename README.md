```python
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist, fashion_mnist
import numpy as np
import matplotlib.pyplot as plt

```

# Separazione di immagini

Questo progetto ha l'obiettivo di separare un'immagine, ottenuta come somma di due immagini, nelle sue componenti originali.

Le due immagini di origine, img1 e img2, provengono da dataset diversi: MNIST e Fashion-MNIST, rispettivamente.

Non è consentita alcuna pre-elaborazione. La rete neurale riceve in input l'immagine combinata (img1 + img2) e restituisce le predizioni (hat_img1 e hat_img2).

Le prestazioni vengono valutate utilizzando l'errore quadratico medio (MSE) tra le immagini predette e quelle di riferimento.

Entrambi i dataset (MNIST e Fashion-MNIST) sono in scala di grigi. Per semplicità, tutti i campioni vengono adattati alla risoluzione (32,32).


```python
(mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test) = mnist.load_data()
print(np.shape(mnist_x_train))
(fashion_mnist_x_train, fashion_mnist_y_train), (fashion_mnist_x_test, fashion_mnist_y_test) = fashion_mnist.load_data()
mnist_x_train = np.pad(mnist_x_train, ((0,0), (2,2), (2,2))) / 255.
print(np.shape(mnist_x_train))
mnist_x_test = np.pad(mnist_x_test, ((0,0), (2,2), (2,2))) / 255.
fashion_mnist_x_train = np.pad(fashion_mnist_x_train, ((0,0), (2,2), (2,2))) / 255.
fashion_mnist_x_test = np.pad(fashion_mnist_x_test, ((0,0), (2,2), (2,2))) / 255.
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
    [1m11490434/11490434[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 0us/step
    (60000, 28, 28)
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz
    [1m29515/29515[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 0us/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
    [1m26421880/26421880[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 0us/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz
    [1m5148/5148[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1us/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz
    [1m4422102/4422102[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 0us/step
    (60000, 32, 32)


#Generazione dei dati


```python
def datagenerator(x1, x2, batchsize):
    n1 = x1.shape[0]
    n2 = x2.shape[0]
    while True:
        num1 = np.random.randint(0, n1, batchsize)
        num2 = np.random.randint(0, n2, batchsize)

        x_data = (x1[num1] + x2[num2]) / 2.0
        y_data = (x1[num1], x2[num2])

        yield x_data, y_data
```

#Modello
Costruisce un modello di rete neurale che:
  - Utilizza immagini 32x32
  - Ha un encoder a blocchi con due convoluzioni ciascuno
  - Ha due decoder (uno per MNIST e uno per Fashion) con skip connections

Le **skip connections** sono collegamenti che passano direttamente informazioni dai livelli iniziali del modello ai livelli più profondi, evitando che vengano completamente perse durante il downsampling.

Il metodo di **He**, o **He Initialization**, è un metodo per inizializzare i pesi di una rete neurale in modo che il flusso del gradiente venga mantenuto stabile durante l'addestramento.
Quando una rete neurale viene creata, i pesi dei neuroni devono essere inizializzati con valori casuali. Un'inizializzazione sbagliata può causare problemi, come:
- Gradiente che esplode (i valori diventano troppo grandi).
- Gradiente che scompare (i valori diventano troppo piccoli, rendendo l'apprendimento lento o impossibile).

Perchè **LeakyReLU**?
Se utilzzassi ReLU alcuni neuroni possono "morire" e non aggiornare mai i loro pesi (smettono di apprendere).
LeakyReLU è una versione migliorata di ReLU, utile per evitare che alcuni neuroni diventino inattivi durante l'addestramento (neurono "morti": neuroni che smettono di apprendere).

Il **bottleneck** è la parte centrale della rete dove i dati vengono compressi prima della ricostruzione. Nel mio modello, serve per creare una rappresentazione compatta prima che i due decoder (MNIST e Fashion MNIST) ricostruiscano le immagini


```python
def build_model():

    # Modifichiamo l'input a 32x32
    inputs = layers.Input(shape=(32, 32))
    x = layers.Reshape((32, 32, 1))(inputs)

    # ----- Encoder -----
    # Blocco 1


    #Crea un layer di convoluzione 2D con 64 filtri.
    #Ogni filtro ha dimensione 3×3
    #padding='same' significa che l’output avrà le stesse dimensioni spaziali dell’input
    #kernel_initializer='he_normal' specifica l’inizializzazione dei pesi secondo il metodo di He.
    conv1 = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x)

    #Applica la normalizzazione per batch (Batch Normalization) all’output della convoluzione.
    #Questa operazione normalizza le attivazioni, migliorando la stabilità e velocizzando l’addestramento.
    conv1 = layers.BatchNormalization()(conv1)


    #applica la funzione di attivazione LeakyReLU con parametro alpha=0.1.
    #A differenza della ReLU standard, LeakyReLU permette a una piccola frazione (qui 10%)
    #dei valori negativi di passare, riducendo il rischio di “morti” dei neuroni.
    conv1 = layers.LeakyReLU(alpha=0.1)(conv1)

    #Applica un’altra convoluzione 2D con gli stessi parametri (64 filtri, kernel 3×3, padding 'same', inizializzazione 'he_normal').
    #Questa convoluzione viene applicata all’output della precedente operazione (già attivato e normalizzato) contenuto in conv1.
    conv1 = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(conv1)


    #Normalizza nuovamente le attivazioni risultanti dalla seconda convoluzione.
    conv1 = layers.BatchNormalization()(conv1)

    #Applica un’altra volta la funzione di attivazione LeakyReLU (con alpha=0.1) all’output normalizzato.
    conv1 = layers.LeakyReLU(alpha=0.1)(conv1)

    #Applica il max pooling con un filtro di dimensione 2x2 all’output di conv1.
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    #Il max pooling riduce le dimensioni spaziali (altezza e larghezza) dimezzandole.
    #Se, ad esempio, l’input aveva dimensioni 32×32, l’output diventerà
    #16×16.
    skip1 = pool1


    # Blocco 2 si comporta in modo simile all'1

    conv2 = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(pool1)

    conv2 = layers.BatchNormalization()(conv2)

    conv2 = layers.LeakyReLU(alpha=0.1)(conv2)

    conv2 = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(conv2)

    conv2 = layers.BatchNormalization()(conv2)

    conv2 = layers.LeakyReLU(alpha=0.1)(conv2)

    pool2 = layers.MaxPooling2D((2, 2))(conv2)  # Output: 8x8

    skip2 = conv2  # Per il decoder MNIST


    # Bottleneck

    # Applica una convoluzione 2D con 256 filtri, kernel 3x3, padding 'same' e inizializzazione 'he_normal'.
    bottleneck = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(pool2)

    # Applica la Batch Normalization per normalizzare l'output e accelerare la convergenza.
    bottleneck = layers.BatchNormalization()(bottleneck)

    # Applica l'attivazione LeakyReLU con alpha=0.1 per introdurre non linearità e gestire i valori negativi.
    bottleneck = layers.LeakyReLU(alpha=0.1)(bottleneck)

    # Applica una Spatial Dropout 2D con tasso di dropout del 30% per ridurre l'overfitting disattivando alcune feature maps.
    bottleneck = layers.SpatialDropout2D(0.3)(bottleneck)



    # Decoder MNIST

    # Applica una convoluzione trasposta (deconvoluzione) per eseguire l'upsampling da 8x8 a 16x16 con 128 filtri.
    up_mnist = layers.Conv2DTranspose(128, (3, 3), strides=2, padding='same', kernel_initializer='he_normal')(bottleneck)

    # Concatena l'output dell'upsampling con la connessione di salto skip2 per recuperare informazioni spaziali perse durante il downsampling.
    up_mnist = layers.concatenate([up_mnist, skip2])

    # Applica una convoluzione 2D con 64 filtri per affinare le caratteristiche dell'immagine ricostruita.
    up_mnist = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(up_mnist)

    # Normalizza l'output della convoluzione per migliorare la stabilità dell'allenamento.
    up_mnist = layers.BatchNormalization()(up_mnist)

    # Applica l'attivazione LeakyReLU con alpha=0.1 per introdurre non linearità.
    up_mnist = layers.LeakyReLU(alpha=0.1)(up_mnist)

    # Applica un dropout del 40% per ridurre l'overfitting.
    up_mnist = layers.Dropout(0.4)(up_mnist)

    # Applica una seconda convoluzione trasposta per eseguire un ulteriore upsampling da 16x16 a 32x32 con 64 filtri.
    up_mnist = layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same', kernel_initializer='he_normal')(up_mnist)

    # Applica una convoluzione finale con 1 filtro e attivazione lineare per ottenere l'output con un singolo canale (scala di grigi).
    mnist_out = layers.Conv2D(1, (3, 3), activation='linear', padding='same', kernel_initializer='he_normal')(up_mnist)

    # Rimodella l'output per ottenere la dimensione finale 32x32.
    mnist_out = layers.Reshape((32, 32), name='mnist_out')(mnist_out)




    # Decoder Fashion

    # Applica una convoluzione trasposta per eseguire l'upsampling da 8x8 a 16x16 con 128 filtri.
    up_fashion = layers.Conv2DTranspose(128, (3, 3), strides=2, padding='same', kernel_initializer='he_normal')(bottleneck)

    # Adatta il tensore di skip1 con una convoluzione 1x1 per ridurre il numero di filtri e renderlo compatibile per la concatenazione.
    skip1_adjusted = layers.Conv2D(64, (1, 1), padding='same', kernel_initializer='he_normal')(skip1)

    # Concatena l'output dell'upsampling con skip1_adjusted per recuperare informazioni perse nel downsampling.
    up_fashion = layers.concatenate([up_fashion, skip1_adjusted])

    # Applica una convoluzione 2D con 64 filtri per affinare le caratteristiche dell'immagine ricostruita.
    up_fashion = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(up_fashion)

    # Normalizza l'output della convoluzione per migliorare la stabilità dell'allenamento.
    up_fashion = layers.BatchNormalization()(up_fashion)

    # Applica l'attivazione LeakyReLU con alpha=0.1 per introdurre non linearità.
    up_fashion = layers.LeakyReLU(alpha=0.1)(up_fashion)

    # Applica una seconda convoluzione trasposta per eseguire un ulteriore upsampling da 16x16 a 32x32 con 64 filtri.
    up_fashion = layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same', kernel_initializer='he_normal')(up_fashion)

    # Applica una convoluzione finale con 1 filtro e attivazione lineare per ottenere l'output con un singolo canale (scala di grigi).
    fashion_out = layers.Conv2D(1, (3, 3), activation='linear', padding='same', kernel_initializer='he_normal')(up_fashion)

    # Rimodella l'output per ottenere la dimensione finale 32x32.
    fashion_out = layers.Reshape((32, 32), name='fashion_out')(fashion_out)

    # Definisce il modello Keras con input iniziale e due output: mnist_out e fashion_out.
    model = tf.keras.Model(inputs, [mnist_out, fashion_out])



    return model

```

#Training



```python
# Definisce l'ottimizzatore Adam con learning rate 0.001 e clipnorm=1.0.
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)

# Costruisce il modello utilizzando la funzione build_model().
model = build_model()

# Compila il modello
model.compile(
    optimizer=optimizer,
    loss={'mnist_out': 'mse', 'fashion_out': 'mse'},
    loss_weights=[1.5, 0.5],
    metrics={'mnist_out': ['mse'], 'fashion_out': ['mse']}
)

# Definisce la dimensione del batch per l'addestramento.
batch_size = 128

# Crea i generatori di dati per il training e la validazione utilizzando la funzione datagenerator().
train_gen = datagenerator(mnist_x_train, fashion_mnist_x_train, batch_size)
val_gen = datagenerator(mnist_x_test, fashion_mnist_x_test, batch_size)

# Definisce una lista di callback per migliorare l'addestramento.
callbacks = [
    # Salva il modello con la minore perdita di validazione.
    tf.keras.callbacks.ModelCheckpoint("best_model_improved.keras", monitor="val_loss", save_best_only=True, verbose=1),

    # Riduce il learning rate di un fattore 0.7 se la perdita di validazione non migliora per 2 epoche.
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.7, patience=2, verbose=1),

    # Interrompe l'addestramento se la perdita di 'mnist_out' non migliora per 4 epoche e ripristina i pesi migliori.
    tf.keras.callbacks.EarlyStopping(monitor="val_mnist_out_loss", patience=4, restore_best_weights=True, mode="min")
]

# Avvia l'addestramento del modello con:
# - Il generatore di dati per il training.
# - 50 epoche massime.
# - 1000 steps_per_epoch.
# - Il generatore di dati per la validazione.
# - 200 passi per la validazione.
# - I callback definiti in precedenza.

history = model.fit(
    train_gen,
    epochs=50,
    steps_per_epoch=1000,
    validation_data=val_gen,
    validation_steps=200,
    callbacks=callbacks,
    verbose=1
)

```

    /usr/local/lib/python3.11/dist-packages/keras/src/layers/activations/leaky_relu.py:41: UserWarning: Argument `alpha` is deprecated. Use `negative_slope` instead.
      warnings.warn(


    Epoch 1/50
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 58ms/step - fashion_out_loss: 0.0299 - fashion_out_mse: 0.0299 - loss: 0.0929 - mnist_out_loss: 0.0520 - mnist_out_mse: 0.0520
    Epoch 1: val_loss improved from inf to 0.01100, saving model to best_model_improved.keras
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m88s[0m 64ms/step - fashion_out_loss: 0.0299 - fashion_out_mse: 0.0299 - loss: 0.0928 - mnist_out_loss: 0.0519 - mnist_out_mse: 0.0519 - val_fashion_out_loss: 0.0064 - val_fashion_out_mse: 0.0064 - val_loss: 0.0110 - val_mnist_out_loss: 0.0052 - val_mnist_out_mse: 0.0052 - learning_rate: 0.0010
    Epoch 2/50
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 59ms/step - fashion_out_loss: 0.0060 - fashion_out_mse: 0.0060 - loss: 0.0122 - mnist_out_loss: 0.0062 - mnist_out_mse: 0.0062
    Epoch 2: val_loss improved from 0.01100 to 0.00747, saving model to best_model_improved.keras
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m64s[0m 64ms/step - fashion_out_loss: 0.0060 - fashion_out_mse: 0.0060 - loss: 0.0122 - mnist_out_loss: 0.0062 - mnist_out_mse: 0.0062 - val_fashion_out_loss: 0.0043 - val_fashion_out_mse: 0.0043 - val_loss: 0.0075 - val_mnist_out_loss: 0.0035 - val_mnist_out_mse: 0.0035 - learning_rate: 0.0010
    Epoch 3/50
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 59ms/step - fashion_out_loss: 0.0044 - fashion_out_mse: 0.0044 - loss: 0.0092 - mnist_out_loss: 0.0047 - mnist_out_mse: 0.0047
    Epoch 3: val_loss improved from 0.00747 to 0.00592, saving model to best_model_improved.keras
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m63s[0m 63ms/step - fashion_out_loss: 0.0044 - fashion_out_mse: 0.0044 - loss: 0.0092 - mnist_out_loss: 0.0047 - mnist_out_mse: 0.0047 - val_fashion_out_loss: 0.0040 - val_fashion_out_mse: 0.0040 - val_loss: 0.0059 - val_mnist_out_loss: 0.0026 - val_mnist_out_mse: 0.0026 - learning_rate: 0.0010
    Epoch 4/50
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 59ms/step - fashion_out_loss: 0.0037 - fashion_out_mse: 0.0037 - loss: 0.0079 - mnist_out_loss: 0.0040 - mnist_out_mse: 0.0040
    Epoch 4: val_loss improved from 0.00592 to 0.00488, saving model to best_model_improved.keras
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m63s[0m 63ms/step - fashion_out_loss: 0.0037 - fashion_out_mse: 0.0037 - loss: 0.0079 - mnist_out_loss: 0.0040 - mnist_out_mse: 0.0040 - val_fashion_out_loss: 0.0028 - val_fashion_out_mse: 0.0028 - val_loss: 0.0049 - val_mnist_out_loss: 0.0023 - val_mnist_out_mse: 0.0023 - learning_rate: 0.0010
    Epoch 5/50
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 59ms/step - fashion_out_loss: 0.0031 - fashion_out_mse: 0.0031 - loss: 0.0070 - mnist_out_loss: 0.0036 - mnist_out_mse: 0.0036
    Epoch 5: val_loss improved from 0.00488 to 0.00487, saving model to best_model_improved.keras
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m82s[0m 82ms/step - fashion_out_loss: 0.0031 - fashion_out_mse: 0.0031 - loss: 0.0070 - mnist_out_loss: 0.0036 - mnist_out_mse: 0.0036 - val_fashion_out_loss: 0.0025 - val_fashion_out_mse: 0.0025 - val_loss: 0.0049 - val_mnist_out_loss: 0.0024 - val_mnist_out_mse: 0.0024 - learning_rate: 0.0010
    Epoch 6/50
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 59ms/step - fashion_out_loss: 0.0027 - fashion_out_mse: 0.0027 - loss: 0.0064 - mnist_out_loss: 0.0034 - mnist_out_mse: 0.0034
    Epoch 6: val_loss improved from 0.00487 to 0.00442, saving model to best_model_improved.keras
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m64s[0m 64ms/step - fashion_out_loss: 0.0027 - fashion_out_mse: 0.0027 - loss: 0.0064 - mnist_out_loss: 0.0034 - mnist_out_mse: 0.0034 - val_fashion_out_loss: 0.0022 - val_fashion_out_mse: 0.0022 - val_loss: 0.0044 - val_mnist_out_loss: 0.0022 - val_mnist_out_mse: 0.0022 - learning_rate: 0.0010
    Epoch 7/50
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 59ms/step - fashion_out_loss: 0.0024 - fashion_out_mse: 0.0024 - loss: 0.0060 - mnist_out_loss: 0.0032 - mnist_out_mse: 0.0032
    Epoch 7: val_loss improved from 0.00442 to 0.00369, saving model to best_model_improved.keras
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m82s[0m 82ms/step - fashion_out_loss: 0.0024 - fashion_out_mse: 0.0024 - loss: 0.0060 - mnist_out_loss: 0.0032 - mnist_out_mse: 0.0032 - val_fashion_out_loss: 0.0022 - val_fashion_out_mse: 0.0022 - val_loss: 0.0037 - val_mnist_out_loss: 0.0017 - val_mnist_out_mse: 0.0017 - learning_rate: 0.0010
    Epoch 8/50
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 59ms/step - fashion_out_loss: 0.0022 - fashion_out_mse: 0.0022 - loss: 0.0057 - mnist_out_loss: 0.0030 - mnist_out_mse: 0.0030
    Epoch 8: val_loss improved from 0.00369 to 0.00355, saving model to best_model_improved.keras
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m64s[0m 64ms/step - fashion_out_loss: 0.0022 - fashion_out_mse: 0.0022 - loss: 0.0057 - mnist_out_loss: 0.0030 - mnist_out_mse: 0.0030 - val_fashion_out_loss: 0.0019 - val_fashion_out_mse: 0.0019 - val_loss: 0.0035 - val_mnist_out_loss: 0.0017 - val_mnist_out_mse: 0.0017 - learning_rate: 0.0010
    Epoch 9/50
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 59ms/step - fashion_out_loss: 0.0020 - fashion_out_mse: 0.0020 - loss: 0.0053 - mnist_out_loss: 0.0029 - mnist_out_mse: 0.0029
    Epoch 9: val_loss improved from 0.00355 to 0.00324, saving model to best_model_improved.keras
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m63s[0m 63ms/step - fashion_out_loss: 0.0020 - fashion_out_mse: 0.0020 - loss: 0.0053 - mnist_out_loss: 0.0029 - mnist_out_mse: 0.0029 - val_fashion_out_loss: 0.0018 - val_fashion_out_mse: 0.0018 - val_loss: 0.0032 - val_mnist_out_loss: 0.0016 - val_mnist_out_mse: 0.0016 - learning_rate: 0.0010
    Epoch 10/50
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 59ms/step - fashion_out_loss: 0.0019 - fashion_out_mse: 0.0019 - loss: 0.0051 - mnist_out_loss: 0.0028 - mnist_out_mse: 0.0028
    Epoch 10: val_loss did not improve from 0.00324
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m82s[0m 82ms/step - fashion_out_loss: 0.0019 - fashion_out_mse: 0.0019 - loss: 0.0051 - mnist_out_loss: 0.0028 - mnist_out_mse: 0.0028 - val_fashion_out_loss: 0.0018 - val_fashion_out_mse: 0.0018 - val_loss: 0.0035 - val_mnist_out_loss: 0.0017 - val_mnist_out_mse: 0.0017 - learning_rate: 0.0010
    Epoch 11/50
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 59ms/step - fashion_out_loss: 0.0018 - fashion_out_mse: 0.0018 - loss: 0.0050 - mnist_out_loss: 0.0027 - mnist_out_mse: 0.0027
    Epoch 11: val_loss improved from 0.00324 to 0.00299, saving model to best_model_improved.keras
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m64s[0m 64ms/step - fashion_out_loss: 0.0018 - fashion_out_mse: 0.0018 - loss: 0.0050 - mnist_out_loss: 0.0027 - mnist_out_mse: 0.0027 - val_fashion_out_loss: 0.0016 - val_fashion_out_mse: 0.0016 - val_loss: 0.0030 - val_mnist_out_loss: 0.0015 - val_mnist_out_mse: 0.0015 - learning_rate: 0.0010
    Epoch 12/50
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 59ms/step - fashion_out_loss: 0.0017 - fashion_out_mse: 0.0017 - loss: 0.0048 - mnist_out_loss: 0.0026 - mnist_out_mse: 0.0026
    Epoch 12: val_loss did not improve from 0.00299
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m82s[0m 82ms/step - fashion_out_loss: 0.0017 - fashion_out_mse: 0.0017 - loss: 0.0048 - mnist_out_loss: 0.0026 - mnist_out_mse: 0.0026 - val_fashion_out_loss: 0.0015 - val_fashion_out_mse: 0.0015 - val_loss: 0.0031 - val_mnist_out_loss: 0.0016 - val_mnist_out_mse: 0.0016 - learning_rate: 0.0010
    Epoch 13/50
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 59ms/step - fashion_out_loss: 0.0017 - fashion_out_mse: 0.0017 - loss: 0.0047 - mnist_out_loss: 0.0026 - mnist_out_mse: 0.0026
    Epoch 13: val_loss did not improve from 0.00299
    
    Epoch 13: ReduceLROnPlateau reducing learning rate to 0.0007000000332482159.
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m63s[0m 63ms/step - fashion_out_loss: 0.0017 - fashion_out_mse: 0.0017 - loss: 0.0047 - mnist_out_loss: 0.0026 - mnist_out_mse: 0.0026 - val_fashion_out_loss: 0.0015 - val_fashion_out_mse: 0.0015 - val_loss: 0.0031 - val_mnist_out_loss: 0.0016 - val_mnist_out_mse: 0.0016 - learning_rate: 0.0010
    Epoch 14/50
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 59ms/step - fashion_out_loss: 0.0015 - fashion_out_mse: 0.0015 - loss: 0.0045 - mnist_out_loss: 0.0025 - mnist_out_mse: 0.0025
    Epoch 14: val_loss improved from 0.00299 to 0.00265, saving model to best_model_improved.keras
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m82s[0m 82ms/step - fashion_out_loss: 0.0015 - fashion_out_mse: 0.0015 - loss: 0.0045 - mnist_out_loss: 0.0025 - mnist_out_mse: 0.0025 - val_fashion_out_loss: 0.0015 - val_fashion_out_mse: 0.0015 - val_loss: 0.0026 - val_mnist_out_loss: 0.0013 - val_mnist_out_mse: 0.0013 - learning_rate: 7.0000e-04
    Epoch 15/50
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 60ms/step - fashion_out_loss: 0.0015 - fashion_out_mse: 0.0015 - loss: 0.0044 - mnist_out_loss: 0.0024 - mnist_out_mse: 0.0024
    Epoch 15: val_loss improved from 0.00265 to 0.00254, saving model to best_model_improved.keras
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m65s[0m 65ms/step - fashion_out_loss: 0.0015 - fashion_out_mse: 0.0015 - loss: 0.0044 - mnist_out_loss: 0.0024 - mnist_out_mse: 0.0024 - val_fashion_out_loss: 0.0014 - val_fashion_out_mse: 0.0014 - val_loss: 0.0025 - val_mnist_out_loss: 0.0012 - val_mnist_out_mse: 0.0012 - learning_rate: 7.0000e-04
    Epoch 16/50
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 60ms/step - fashion_out_loss: 0.0015 - fashion_out_mse: 0.0015 - loss: 0.0044 - mnist_out_loss: 0.0024 - mnist_out_mse: 0.0024
    Epoch 16: val_loss improved from 0.00254 to 0.00253, saving model to best_model_improved.keras
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m82s[0m 82ms/step - fashion_out_loss: 0.0015 - fashion_out_mse: 0.0015 - loss: 0.0044 - mnist_out_loss: 0.0024 - mnist_out_mse: 0.0024 - val_fashion_out_loss: 0.0013 - val_fashion_out_mse: 0.0013 - val_loss: 0.0025 - val_mnist_out_loss: 0.0012 - val_mnist_out_mse: 0.0012 - learning_rate: 7.0000e-04
    Epoch 17/50
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 60ms/step - fashion_out_loss: 0.0014 - fashion_out_mse: 0.0014 - loss: 0.0043 - mnist_out_loss: 0.0024 - mnist_out_mse: 0.0024
    Epoch 17: val_loss did not improve from 0.00253
    
    Epoch 17: ReduceLROnPlateau reducing learning rate to 0.0004900000232737511.
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m82s[0m 82ms/step - fashion_out_loss: 0.0014 - fashion_out_mse: 0.0014 - loss: 0.0043 - mnist_out_loss: 0.0024 - mnist_out_mse: 0.0024 - val_fashion_out_loss: 0.0013 - val_fashion_out_mse: 0.0013 - val_loss: 0.0025 - val_mnist_out_loss: 0.0013 - val_mnist_out_mse: 0.0013 - learning_rate: 7.0000e-04
    Epoch 18/50
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 60ms/step - fashion_out_loss: 0.0014 - fashion_out_mse: 0.0014 - loss: 0.0042 - mnist_out_loss: 0.0023 - mnist_out_mse: 0.0023
    Epoch 18: val_loss improved from 0.00253 to 0.00232, saving model to best_model_improved.keras
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m82s[0m 82ms/step - fashion_out_loss: 0.0014 - fashion_out_mse: 0.0014 - loss: 0.0042 - mnist_out_loss: 0.0023 - mnist_out_mse: 0.0023 - val_fashion_out_loss: 0.0013 - val_fashion_out_mse: 0.0013 - val_loss: 0.0023 - val_mnist_out_loss: 0.0011 - val_mnist_out_mse: 0.0011 - learning_rate: 4.9000e-04
    Epoch 19/50
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 60ms/step - fashion_out_loss: 0.0013 - fashion_out_mse: 0.0013 - loss: 0.0041 - mnist_out_loss: 0.0023 - mnist_out_mse: 0.0023
    Epoch 19: val_loss did not improve from 0.00232
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m82s[0m 82ms/step - fashion_out_loss: 0.0013 - fashion_out_mse: 0.0013 - loss: 0.0041 - mnist_out_loss: 0.0023 - mnist_out_mse: 0.0023 - val_fashion_out_loss: 0.0012 - val_fashion_out_mse: 0.0012 - val_loss: 0.0024 - val_mnist_out_loss: 0.0012 - val_mnist_out_mse: 0.0012 - learning_rate: 4.9000e-04
    Epoch 20/50
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 60ms/step - fashion_out_loss: 0.0013 - fashion_out_mse: 0.0013 - loss: 0.0041 - mnist_out_loss: 0.0023 - mnist_out_mse: 0.0023
    Epoch 20: val_loss improved from 0.00232 to 0.00228, saving model to best_model_improved.keras
    
    Epoch 20: ReduceLROnPlateau reducing learning rate to 0.00034300000406801696.
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m64s[0m 64ms/step - fashion_out_loss: 0.0013 - fashion_out_mse: 0.0013 - loss: 0.0041 - mnist_out_loss: 0.0023 - mnist_out_mse: 0.0023 - val_fashion_out_loss: 0.0012 - val_fashion_out_mse: 0.0012 - val_loss: 0.0023 - val_mnist_out_loss: 0.0011 - val_mnist_out_mse: 0.0011 - learning_rate: 4.9000e-04
    Epoch 21/50
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 61ms/step - fashion_out_loss: 0.0013 - fashion_out_mse: 0.0013 - loss: 0.0041 - mnist_out_loss: 0.0023 - mnist_out_mse: 0.0023
    Epoch 21: val_loss did not improve from 0.00228
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m82s[0m 82ms/step - fashion_out_loss: 0.0013 - fashion_out_mse: 0.0013 - loss: 0.0041 - mnist_out_loss: 0.0023 - mnist_out_mse: 0.0023 - val_fashion_out_loss: 0.0012 - val_fashion_out_mse: 0.0012 - val_loss: 0.0024 - val_mnist_out_loss: 0.0012 - val_mnist_out_mse: 0.0012 - learning_rate: 3.4300e-04
    Epoch 22/50
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 60ms/step - fashion_out_loss: 0.0013 - fashion_out_mse: 0.0013 - loss: 0.0040 - mnist_out_loss: 0.0022 - mnist_out_mse: 0.0022
    Epoch 22: val_loss improved from 0.00228 to 0.00224, saving model to best_model_improved.keras
    
    Epoch 22: ReduceLROnPlateau reducing learning rate to 0.00024009999469853935.
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m82s[0m 82ms/step - fashion_out_loss: 0.0013 - fashion_out_mse: 0.0013 - loss: 0.0040 - mnist_out_loss: 0.0022 - mnist_out_mse: 0.0022 - val_fashion_out_loss: 0.0012 - val_fashion_out_mse: 0.0012 - val_loss: 0.0022 - val_mnist_out_loss: 0.0011 - val_mnist_out_mse: 0.0011 - learning_rate: 3.4300e-04
    Epoch 23/50
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 60ms/step - fashion_out_loss: 0.0012 - fashion_out_mse: 0.0012 - loss: 0.0039 - mnist_out_loss: 0.0022 - mnist_out_mse: 0.0022
    Epoch 23: val_loss improved from 0.00224 to 0.00218, saving model to best_model_improved.keras
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m82s[0m 82ms/step - fashion_out_loss: 0.0012 - fashion_out_mse: 0.0012 - loss: 0.0039 - mnist_out_loss: 0.0022 - mnist_out_mse: 0.0022 - val_fashion_out_loss: 0.0011 - val_fashion_out_mse: 0.0011 - val_loss: 0.0022 - val_mnist_out_loss: 0.0011 - val_mnist_out_mse: 0.0011 - learning_rate: 2.4010e-04
    Epoch 24/50
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 60ms/step - fashion_out_loss: 0.0012 - fashion_out_mse: 0.0012 - loss: 0.0039 - mnist_out_loss: 0.0022 - mnist_out_mse: 0.0022
    Epoch 24: val_loss did not improve from 0.00218
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m82s[0m 82ms/step - fashion_out_loss: 0.0012 - fashion_out_mse: 0.0012 - loss: 0.0039 - mnist_out_loss: 0.0022 - mnist_out_mse: 0.0022 - val_fashion_out_loss: 0.0011 - val_fashion_out_mse: 0.0011 - val_loss: 0.0022 - val_mnist_out_loss: 0.0011 - val_mnist_out_mse: 0.0011 - learning_rate: 2.4010e-04
    Epoch 25/50
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 60ms/step - fashion_out_loss: 0.0012 - fashion_out_mse: 0.0012 - loss: 0.0039 - mnist_out_loss: 0.0022 - mnist_out_mse: 0.0022
    Epoch 25: val_loss improved from 0.00218 to 0.00214, saving model to best_model_improved.keras
    
    Epoch 25: ReduceLROnPlateau reducing learning rate to 0.00016806999628897755.
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m82s[0m 82ms/step - fashion_out_loss: 0.0012 - fashion_out_mse: 0.0012 - loss: 0.0039 - mnist_out_loss: 0.0022 - mnist_out_mse: 0.0022 - val_fashion_out_loss: 0.0011 - val_fashion_out_mse: 0.0011 - val_loss: 0.0021 - val_mnist_out_loss: 0.0010 - val_mnist_out_mse: 0.0010 - learning_rate: 2.4010e-04
    Epoch 26/50
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 60ms/step - fashion_out_loss: 0.0012 - fashion_out_mse: 0.0012 - loss: 0.0039 - mnist_out_loss: 0.0022 - mnist_out_mse: 0.0022
    Epoch 26: val_loss improved from 0.00214 to 0.00213, saving model to best_model_improved.keras
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m64s[0m 65ms/step - fashion_out_loss: 0.0012 - fashion_out_mse: 0.0012 - loss: 0.0039 - mnist_out_loss: 0.0022 - mnist_out_mse: 0.0022 - val_fashion_out_loss: 0.0011 - val_fashion_out_mse: 0.0011 - val_loss: 0.0021 - val_mnist_out_loss: 0.0010 - val_mnist_out_mse: 0.0010 - learning_rate: 1.6807e-04
    Epoch 27/50
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 60ms/step - fashion_out_loss: 0.0012 - fashion_out_mse: 0.0012 - loss: 0.0038 - mnist_out_loss: 0.0022 - mnist_out_mse: 0.0022
    Epoch 27: val_loss did not improve from 0.00213
    
    Epoch 27: ReduceLROnPlateau reducing learning rate to 0.00011764899536501615.
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m82s[0m 82ms/step - fashion_out_loss: 0.0012 - fashion_out_mse: 0.0012 - loss: 0.0038 - mnist_out_loss: 0.0022 - mnist_out_mse: 0.0022 - val_fashion_out_loss: 0.0011 - val_fashion_out_mse: 0.0011 - val_loss: 0.0022 - val_mnist_out_loss: 0.0011 - val_mnist_out_mse: 0.0011 - learning_rate: 1.6807e-04
    Epoch 28/50
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 60ms/step - fashion_out_loss: 0.0012 - fashion_out_mse: 0.0012 - loss: 0.0038 - mnist_out_loss: 0.0022 - mnist_out_mse: 0.0022
    Epoch 28: val_loss improved from 0.00213 to 0.00209, saving model to best_model_improved.keras
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m82s[0m 82ms/step - fashion_out_loss: 0.0012 - fashion_out_mse: 0.0012 - loss: 0.0038 - mnist_out_loss: 0.0022 - mnist_out_mse: 0.0022 - val_fashion_out_loss: 0.0011 - val_fashion_out_mse: 0.0011 - val_loss: 0.0021 - val_mnist_out_loss: 0.0010 - val_mnist_out_mse: 0.0010 - learning_rate: 1.1765e-04
    Epoch 29/50
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 60ms/step - fashion_out_loss: 0.0012 - fashion_out_mse: 0.0012 - loss: 0.0038 - mnist_out_loss: 0.0021 - mnist_out_mse: 0.0021
    Epoch 29: val_loss did not improve from 0.00209
    
    Epoch 29: ReduceLROnPlateau reducing learning rate to 8.235429777414538e-05.
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m82s[0m 82ms/step - fashion_out_loss: 0.0012 - fashion_out_mse: 0.0012 - loss: 0.0038 - mnist_out_loss: 0.0021 - mnist_out_mse: 0.0021 - val_fashion_out_loss: 0.0011 - val_fashion_out_mse: 0.0011 - val_loss: 0.0021 - val_mnist_out_loss: 0.0010 - val_mnist_out_mse: 0.0010 - learning_rate: 1.1765e-04
    Epoch 30/50
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 60ms/step - fashion_out_loss: 0.0012 - fashion_out_mse: 0.0012 - loss: 0.0038 - mnist_out_loss: 0.0021 - mnist_out_mse: 0.0021
    Epoch 30: val_loss did not improve from 0.00209
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m64s[0m 64ms/step - fashion_out_loss: 0.0012 - fashion_out_mse: 0.0012 - loss: 0.0038 - mnist_out_loss: 0.0021 - mnist_out_mse: 0.0021 - val_fashion_out_loss: 0.0011 - val_fashion_out_mse: 0.0011 - val_loss: 0.0021 - val_mnist_out_loss: 0.0011 - val_mnist_out_mse: 0.0011 - learning_rate: 8.2354e-05
    Epoch 31/50
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 60ms/step - fashion_out_loss: 0.0012 - fashion_out_mse: 0.0012 - loss: 0.0038 - mnist_out_loss: 0.0021 - mnist_out_mse: 0.0021
    Epoch 31: val_loss improved from 0.00209 to 0.00207, saving model to best_model_improved.keras
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m64s[0m 64ms/step - fashion_out_loss: 0.0012 - fashion_out_mse: 0.0012 - loss: 0.0038 - mnist_out_loss: 0.0021 - mnist_out_mse: 0.0021 - val_fashion_out_loss: 0.0011 - val_fashion_out_mse: 0.0011 - val_loss: 0.0021 - val_mnist_out_loss: 0.0010 - val_mnist_out_mse: 0.0010 - learning_rate: 8.2354e-05
    Epoch 32/50
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 60ms/step - fashion_out_loss: 0.0012 - fashion_out_mse: 0.0012 - loss: 0.0038 - mnist_out_loss: 0.0021 - mnist_out_mse: 0.0021
    Epoch 32: val_loss improved from 0.00207 to 0.00206, saving model to best_model_improved.keras
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m82s[0m 82ms/step - fashion_out_loss: 0.0012 - fashion_out_mse: 0.0012 - loss: 0.0038 - mnist_out_loss: 0.0021 - mnist_out_mse: 0.0021 - val_fashion_out_loss: 0.0011 - val_fashion_out_mse: 0.0011 - val_loss: 0.0021 - val_mnist_out_loss: 0.0010 - val_mnist_out_mse: 0.0010 - learning_rate: 8.2354e-05
    Epoch 33/50
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 60ms/step - fashion_out_loss: 0.0012 - fashion_out_mse: 0.0012 - loss: 0.0038 - mnist_out_loss: 0.0021 - mnist_out_mse: 0.0021
    Epoch 33: val_loss improved from 0.00206 to 0.00206, saving model to best_model_improved.keras
    
    Epoch 33: ReduceLROnPlateau reducing learning rate to 5.76480058953166e-05.
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m82s[0m 82ms/step - fashion_out_loss: 0.0012 - fashion_out_mse: 0.0012 - loss: 0.0038 - mnist_out_loss: 0.0021 - mnist_out_mse: 0.0021 - val_fashion_out_loss: 0.0011 - val_fashion_out_mse: 0.0011 - val_loss: 0.0021 - val_mnist_out_loss: 0.0010 - val_mnist_out_mse: 0.0010 - learning_rate: 8.2354e-05
    Epoch 34/50
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 61ms/step - fashion_out_loss: 0.0011 - fashion_out_mse: 0.0011 - loss: 0.0038 - mnist_out_loss: 0.0021 - mnist_out_mse: 0.0021
    Epoch 34: val_loss did not improve from 0.00206
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m65s[0m 65ms/step - fashion_out_loss: 0.0011 - fashion_out_mse: 0.0011 - loss: 0.0038 - mnist_out_loss: 0.0021 - mnist_out_mse: 0.0021 - val_fashion_out_loss: 0.0011 - val_fashion_out_mse: 0.0011 - val_loss: 0.0021 - val_mnist_out_loss: 0.0010 - val_mnist_out_mse: 0.0010 - learning_rate: 5.7648e-05
    Epoch 35/50
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 60ms/step - fashion_out_loss: 0.0011 - fashion_out_mse: 0.0011 - loss: 0.0038 - mnist_out_loss: 0.0021 - mnist_out_mse: 0.0021
    Epoch 35: val_loss improved from 0.00206 to 0.00202, saving model to best_model_improved.keras
    
    Epoch 35: ReduceLROnPlateau reducing learning rate to 4.0353603617404586e-05.
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m65s[0m 65ms/step - fashion_out_loss: 0.0011 - fashion_out_mse: 0.0011 - loss: 0.0038 - mnist_out_loss: 0.0021 - mnist_out_mse: 0.0021 - val_fashion_out_loss: 0.0011 - val_fashion_out_mse: 0.0011 - val_loss: 0.0020 - val_mnist_out_loss: 9.9494e-04 - val_mnist_out_mse: 9.9494e-04 - learning_rate: 5.7648e-05
    Epoch 36/50
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 60ms/step - fashion_out_loss: 0.0011 - fashion_out_mse: 0.0011 - loss: 0.0037 - mnist_out_loss: 0.0021 - mnist_out_mse: 0.0021
    Epoch 36: val_loss improved from 0.00202 to 0.00202, saving model to best_model_improved.keras
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m82s[0m 82ms/step - fashion_out_loss: 0.0011 - fashion_out_mse: 0.0011 - loss: 0.0037 - mnist_out_loss: 0.0021 - mnist_out_mse: 0.0021 - val_fashion_out_loss: 0.0011 - val_fashion_out_mse: 0.0011 - val_loss: 0.0020 - val_mnist_out_loss: 9.9275e-04 - val_mnist_out_mse: 9.9275e-04 - learning_rate: 4.0354e-05
    Epoch 37/50
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 60ms/step - fashion_out_loss: 0.0011 - fashion_out_mse: 0.0011 - loss: 0.0037 - mnist_out_loss: 0.0021 - mnist_out_mse: 0.0021
    Epoch 37: val_loss did not improve from 0.00202
    
    Epoch 37: ReduceLROnPlateau reducing learning rate to 2.8247522277524694e-05.
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m82s[0m 82ms/step - fashion_out_loss: 0.0011 - fashion_out_mse: 0.0011 - loss: 0.0037 - mnist_out_loss: 0.0021 - mnist_out_mse: 0.0021 - val_fashion_out_loss: 0.0011 - val_fashion_out_mse: 0.0011 - val_loss: 0.0020 - val_mnist_out_loss: 9.9893e-04 - val_mnist_out_mse: 9.9893e-04 - learning_rate: 4.0354e-05
    Epoch 38/50
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 60ms/step - fashion_out_loss: 0.0011 - fashion_out_mse: 0.0011 - loss: 0.0037 - mnist_out_loss: 0.0021 - mnist_out_mse: 0.0021
    Epoch 38: val_loss did not improve from 0.00202
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m64s[0m 64ms/step - fashion_out_loss: 0.0011 - fashion_out_mse: 0.0011 - loss: 0.0037 - mnist_out_loss: 0.0021 - mnist_out_mse: 0.0021 - val_fashion_out_loss: 0.0011 - val_fashion_out_mse: 0.0011 - val_loss: 0.0020 - val_mnist_out_loss: 9.9948e-04 - val_mnist_out_mse: 9.9948e-04 - learning_rate: 2.8248e-05
    Epoch 39/50
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 60ms/step - fashion_out_loss: 0.0011 - fashion_out_mse: 0.0011 - loss: 0.0037 - mnist_out_loss: 0.0021 - mnist_out_mse: 0.0021
    Epoch 39: val_loss did not improve from 0.00202
    
    Epoch 39: ReduceLROnPlateau reducing learning rate to 1.977326610358432e-05.
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m82s[0m 82ms/step - fashion_out_loss: 0.0011 - fashion_out_mse: 0.0011 - loss: 0.0037 - mnist_out_loss: 0.0021 - mnist_out_mse: 0.0021 - val_fashion_out_loss: 0.0011 - val_fashion_out_mse: 0.0011 - val_loss: 0.0020 - val_mnist_out_loss: 9.9918e-04 - val_mnist_out_mse: 9.9918e-04 - learning_rate: 2.8248e-05
    Epoch 40/50
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 60ms/step - fashion_out_loss: 0.0011 - fashion_out_mse: 0.0011 - loss: 0.0037 - mnist_out_loss: 0.0021 - mnist_out_mse: 0.0021
    Epoch 40: val_loss improved from 0.00202 to 0.00202, saving model to best_model_improved.keras
    [1m1000/1000[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m65s[0m 65ms/step - fashion_out_loss: 0.0011 - fashion_out_mse: 0.0011 - loss: 0.0037 - mnist_out_loss: 0.0021 - mnist_out_mse: 0.0021 - val_fashion_out_loss: 0.0011 - val_fashion_out_mse: 0.0011 - val_loss: 0.0020 - val_mnist_out_loss: 9.9281e-04 - val_mnist_out_mse: 9.9281e-04 - learning_rate: 1.9773e-05


#Valutazione


```python
#Training vs Validation Loss

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.show()

```


    
![png](ProgettoApprendimentoAutomatico_files/ProgettoApprendimentoAutomatico_10_0.png)
    



```python
testgen = datagenerator(mnist_x_test,fashion_mnist_x_test,1000)

eval_samples_x, (eval_samples_y1, eval_sample2) = next(testgen)

def eval_model(model):
  with tf.device('/CPU:0'):
      x, (y1,y2) = next(testgen)

      # use model.predict to get predictions. Here we just call model
      pred1,pred2 = model(x)

  return (np.mean((pred1-y1)**2) + np.mean((pred2-y2)**2) / 2)


repeat_eval = 10
eval_results = []

for i in range(repeat_eval):
  eval_results.append(eval_model(model))

print("mse = ", np.mean(eval_results))
print("standard deviation = ", np.std(eval_results))

```

    mse =  0.0015147798229008913
    standard deviation =  3.9415825320704385e-05



```python
def compute_pixel_accuracy(y_true, y_pred, threshold=0.01):
    """
    Calcola la percentuale di pixel in cui la differenza assoluta
    tra y_true e y_pred è inferiore a threshold.
    """
    # Calcola la differenza assoluta per ogni pixel
    diff = np.abs(y_true - y_pred)
    # Conta i pixel "corretti"
    correct_pixels = diff < threshold
    # Restituisce la media (percentuale di pixel corretti)
    return np.mean(correct_pixels)


def evaluate_accuracy(model, data_gen, steps=200, threshold=0.01):
    """
    Valuta l'"accuracy" del modello su un numero di batch presi dal data generator.
    Ritorna l'accuracy media per il ramo MNIST e per il ramo Fashion.
    """
    acc_mnist_list = []
    acc_fashion_list = []

    for _ in range(steps):
        x, (y_mnist, y_fashion) = next(data_gen)
        # Ottieni le ricostruzioni dal modello
        y_mnist_pred, y_fashion_pred = model.predict(x, verbose=0)
        # Calcola l'accuracy per ciascun ramo
        acc_mnist = compute_pixel_accuracy(y_mnist, y_mnist_pred, threshold)
        acc_fashion = compute_pixel_accuracy(y_fashion, y_fashion_pred, threshold)
        acc_mnist_list.append(acc_mnist)
        acc_fashion_list.append(acc_fashion)

    mean_acc_mnist = np.mean(acc_mnist_list)
    mean_acc_fashion = np.mean(acc_fashion_list)

    return mean_acc_mnist, mean_acc_fashion


val_gen_for_accuracy = datagenerator(mnist_x_test, fashion_mnist_x_test, batchsize=100)
acc_mnist, acc_fashion = evaluate_accuracy(model, val_gen_for_accuracy, steps=200, threshold=0.1)

print("Accuracy media (MNIST):", acc_mnist)
print("Accuracy media (Fashion):", acc_fashion)
print("Accuracy media complessiva:", (acc_mnist + acc_fashion) / 2)

```

    Accuracy media (MNIST): 0.9802001464843749
    Accuracy media (Fashion): 0.981263330078125
    Accuracy media complessiva: 0.9807317382812499



```python
def plot_predictions(model, num_examples=5):
    """Visualizza input combinato e ricostruzioni"""
    # Utilizziamo il dataset corretto: fashion_mnist_x_test
    test_gen = datagenerator(mnist_x_test, fashion_mnist_x_test, 1)

    for _ in range(num_examples):
        x, (y1_true, y2_true) = next(test_gen)
        y1_pred, y2_pred = model.predict(x, verbose=0)

        plt.figure(figsize=(16, 4))
        # Input combinato
        plt.subplot(1, 5, 1)
        plt.imshow(x[0], cmap='gray')
        plt.title('Input')
        plt.axis('off')
        # MNIST Reale
        plt.subplot(1, 5, 2)
        plt.imshow(y1_true[0], cmap='gray')
        plt.title('MNIST Reale')
        plt.axis('off')
        # MNIST Predetto
        plt.subplot(1, 5, 3)
        plt.imshow(y1_pred[0], cmap='gray')
        plt.title('MNIST Predetto')
        plt.axis('off')
        # Fashion Reale
        plt.subplot(1, 5, 4)
        plt.imshow(y2_true[0], cmap='gray')
        plt.title('Fashion Reale')
        plt.axis('off')
        # Fashion Predetto
        plt.subplot(1, 5, 5)
        plt.imshow(y2_pred[0], cmap='gray')
        plt.title('Fashion Predetto')
        plt.axis('off')
        plt.show()

# Mostra esempi
plot_predictions(model)

```


    
![png](ProgettoApprendimentoAutomatico_files/ProgettoApprendimentoAutomatico_13_0.png)
    



    
![png](ProgettoApprendimentoAutomatico_files/ProgettoApprendimentoAutomatico_13_1.png)
    



    
![png](ProgettoApprendimentoAutomatico_files/ProgettoApprendimentoAutomatico_13_2.png)
    



    
![png](ProgettoApprendimentoAutomatico_files/ProgettoApprendimentoAutomatico_13_3.png)
    



    
![png](ProgettoApprendimentoAutomatico_files/ProgettoApprendimentoAutomatico_13_4.png)
    



```python
print(model.summary())
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "functional"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)              </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">        Param # </span>┃<span style="font-weight: bold"> Connected to           </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ input_layer (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)         │              <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ -                      │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ reshape (<span style="color: #0087ff; text-decoration-color: #0087ff">Reshape</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)      │              <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ input_layer[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ conv2d (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)     │            <span style="color: #00af00; text-decoration-color: #00af00">640</span> │ reshape[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]          │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ batch_normalization       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)     │            <span style="color: #00af00; text-decoration-color: #00af00">256</span> │ conv2d[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]           │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)      │                        │                │                        │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ leaky_re_lu (<span style="color: #0087ff; text-decoration-color: #0087ff">LeakyReLU</span>)   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)     │              <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalization[<span style="color: #00af00; text-decoration-color: #00af00">0</span>… │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ conv2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)     │         <span style="color: #00af00; text-decoration-color: #00af00">36,928</span> │ leaky_re_lu[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ batch_normalization_1     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)     │            <span style="color: #00af00; text-decoration-color: #00af00">256</span> │ conv2d_1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]         │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)      │                        │                │                        │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ leaky_re_lu_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">LeakyReLU</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)     │              <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalization_1… │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ max_pooling2d             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)     │              <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ leaky_re_lu_1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)            │                        │                │                        │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ conv2d_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │         <span style="color: #00af00; text-decoration-color: #00af00">73,856</span> │ max_pooling2d[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ batch_normalization_2     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │            <span style="color: #00af00; text-decoration-color: #00af00">512</span> │ conv2d_2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]         │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)      │                        │                │                        │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ leaky_re_lu_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">LeakyReLU</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │              <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalization_2… │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ conv2d_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │        <span style="color: #00af00; text-decoration-color: #00af00">147,584</span> │ leaky_re_lu_2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ batch_normalization_3     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │            <span style="color: #00af00; text-decoration-color: #00af00">512</span> │ conv2d_3[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]         │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)      │                        │                │                        │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ leaky_re_lu_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">LeakyReLU</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │              <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalization_3… │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ max_pooling2d_1           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)      │              <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ leaky_re_lu_3[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)            │                        │                │                        │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ conv2d_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)      │        <span style="color: #00af00; text-decoration-color: #00af00">295,168</span> │ max_pooling2d_1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]  │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ batch_normalization_4     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)      │          <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ conv2d_4[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]         │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)      │                        │                │                        │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ leaky_re_lu_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">LeakyReLU</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)      │              <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalization_4… │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ spatial_dropout2d         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)      │              <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ leaky_re_lu_4[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">SpatialDropout2D</span>)        │                        │                │                        │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ conv2d_transpose          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │        <span style="color: #00af00; text-decoration-color: #00af00">295,040</span> │ spatial_dropout2d[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">…</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2DTranspose</span>)         │                        │                │                        │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ concatenate (<span style="color: #0087ff; text-decoration-color: #0087ff">Concatenate</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │              <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv2d_transpose[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>… │
│                           │                        │                │ leaky_re_lu_3[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ conv2d_transpose_2        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │        <span style="color: #00af00; text-decoration-color: #00af00">295,040</span> │ spatial_dropout2d[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">…</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2DTranspose</span>)         │                        │                │                        │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ conv2d_7 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)     │          <span style="color: #00af00; text-decoration-color: #00af00">4,160</span> │ max_pooling2d[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ conv2d_5 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)     │        <span style="color: #00af00; text-decoration-color: #00af00">147,520</span> │ concatenate[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ concatenate_1             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">192</span>)    │              <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv2d_transpose_2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Concatenate</span>)             │                        │                │ conv2d_7[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]         │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ batch_normalization_5     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)     │            <span style="color: #00af00; text-decoration-color: #00af00">256</span> │ conv2d_5[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]         │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)      │                        │                │                        │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ conv2d_8 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)     │        <span style="color: #00af00; text-decoration-color: #00af00">110,656</span> │ concatenate_1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ leaky_re_lu_5 (<span style="color: #0087ff; text-decoration-color: #0087ff">LeakyReLU</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)     │              <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalization_5… │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ batch_normalization_6     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)     │            <span style="color: #00af00; text-decoration-color: #00af00">256</span> │ conv2d_8[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]         │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)      │                        │                │                        │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)     │              <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ leaky_re_lu_5[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ leaky_re_lu_6 (<span style="color: #0087ff; text-decoration-color: #0087ff">LeakyReLU</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)     │              <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalization_6… │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ conv2d_transpose_1        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)     │         <span style="color: #00af00; text-decoration-color: #00af00">36,928</span> │ dropout[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]          │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2DTranspose</span>)         │                        │                │                        │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ conv2d_transpose_3        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)     │         <span style="color: #00af00; text-decoration-color: #00af00">36,928</span> │ leaky_re_lu_6[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2DTranspose</span>)         │                        │                │                        │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ conv2d_6 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)      │            <span style="color: #00af00; text-decoration-color: #00af00">577</span> │ conv2d_transpose_1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ conv2d_9 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)      │            <span style="color: #00af00; text-decoration-color: #00af00">577</span> │ conv2d_transpose_3[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ mnist_out (<span style="color: #0087ff; text-decoration-color: #0087ff">Reshape</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)         │              <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv2d_6[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]         │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ fashion_out (<span style="color: #0087ff; text-decoration-color: #0087ff">Reshape</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)         │              <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv2d_9[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]         │
└───────────────────────────┴────────────────────────┴────────────────┴────────────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">4,450,952</span> (16.98 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">1,483,138</span> (5.66 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">1,536</span> (6.00 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Optimizer params: </span><span style="color: #00af00; text-decoration-color: #00af00">2,966,278</span> (11.32 MB)
</pre>



    None
