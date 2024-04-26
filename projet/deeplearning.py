
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),  # Pour éviter le surapprentissage
        Dense(num_classes, activation='softmax')
    ])

    return model


image_height = 200
image_width = 200
num_channels =  3

# Définition des dimensions des images d'entrée et du nombre de classes à classifier
input_shape = (image_height, image_width, num_channels)  # Remplacez ces valeurs par les dimensions de vos images
num_classes = 6  # Remplacez par le nombre de gestes que vous voulez classifier

# Construction du modèle CNN
model = build_cnn_model(input_shape, num_classes)

# Compilation du modèle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Affichage de la structure du modèle
print(model.summary())