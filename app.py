import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate

# Ustawienia ścieżek i parametrów
GOOD_DIR = 'train_images/good'
BAD_DIR = 'train_images/bad'
SEG_MODEL_PATH = 'difference_model.h5'
TARGET_SIZE = (224, 224)

# Utworzenie folderów, jeśli nie istnieją
os.makedirs(GOOD_DIR, exist_ok=True)
os.makedirs(BAD_DIR, exist_ok=True)

##############################################
# Funkcja zapisu przesłanego pliku
def save_uploaded_file(uploaded_file, folder):
    file_path = os.path.join(folder, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

##############################################
# Funkcja wyboru referencyjnego zdjęcia z folderu 'train_images/good'
def get_reference_image(target_size=TARGET_SIZE):
    files = os.listdir(GOOD_DIR)
    if not files:
        return None, None
    ref_path = os.path.join(GOOD_DIR, files[0])
    img = load_img(ref_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    return ref_path, img_array

##############################################
# Definicja modelu U-Net do segmentacji różnic
def unet_model(input_size=(224, 224, 6)):
    inputs = Input(input_size)
    
    # Encoder
    conv1 = Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D((2,2))(conv1)
    
    conv2 = Conv2D(128, (3,3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D((2,2))(conv2)
    
    # Bottleneck
    conv3 = Conv2D(256, (3,3), activation='relu', padding='same')(pool2)
    
    # Decoder
    up2 = UpSampling2D((2,2))(conv3)
    concat2 = Concatenate()([up2, conv2])
    conv4 = Conv2D(128, (3,3), activation='relu', padding='same')(concat2)
    
    up1 = UpSampling2D((2,2))(conv4)
    concat1 = Concatenate()([up1, conv1])
    conv5 = Conv2D(64, (3,3), activation='relu', padding='same')(concat1)
    
    outputs = Conv2D(1, (1,1), activation='sigmoid')(conv5)
    model = Model(inputs, outputs)
    return model

##############################################
# Funkcja generująca dane treningowe
def create_training_data(target_size=TARGET_SIZE, thresh_value=70):
    X_list = []
    Y_list = []
    # Pobierz referencyjny obraz z folderu dobrych zdjęć
    ref_path, ref_img_array = get_reference_image(target_size)
    if ref_img_array is None:
        st.error("Brak zdjęć referencyjnych w folderze 'train_images/good'.")
        return None, None
    
    # Dla każdego obrazu z folderu BAD_DIR
    for filename in os.listdir(BAD_DIR):
        file_path = os.path.join(BAD_DIR, filename)
        try:
            bad_img = load_img(file_path, target_size=target_size)
            bad_img_array = img_to_array(bad_img) / 255.0
            
            # Utwórz ground truth maskę na podstawie różnicy między referencyjnym a złym obrazem.
            ref_gray = cv2.cvtColor((ref_img_array*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            bad_gray = cv2.cvtColor((bad_img_array*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            diff = cv2.absdiff(ref_gray, bad_gray)
            # Ustaw próg – wartość progowa może być dostosowywana
            _, mask = cv2.threshold(diff, thresh_value, 255, cv2.THRESH_BINARY)
            # Operacje morfologiczne, aby usunąć drobne szumy
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            mask = mask / 255.0
            mask = np.expand_dims(mask, axis=-1)
            
            # Wejście: sklej referencyjny obraz i zły obraz (kanały: 3+3=6)
            input_pair = np.concatenate([ref_img_array, bad_img_array], axis=-1)
            X_list.append(input_pair)
            Y_list.append(mask)
        except Exception as e:
            st.write(f"Problem z wczytaniem {filename}: {e}")
    if len(X_list) == 0:
        return None, None
    X = np.array(X_list)
    Y = np.array(Y_list)
    return X, Y

##############################################
# Funkcja trenowania modelu segmentacyjnego
def train_segmentation_model():
    tf.keras.backend.clear_session()  # Czyszczenie poprzednich modeli
    X, Y = create_training_data()
    if X is None or Y is None:
        st.error("Nie udało się utworzyć danych treningowych.")
        return None
    model = unet_model(input_size=(TARGET_SIZE[0], TARGET_SIZE[1], 6))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    st.write("Trenowanie modelu segmentacyjnego – proszę czekać...")
    model.fit(X, Y,batch_size=16, epochs=100, validation_split=0.4, verbose=1)
    model.save(SEG_MODEL_PATH)
    st.success("Model segmentacyjny został wytrenowany i zapisany!")
    return model

##############################################
# Funkcja analizy nowego zdjęcia przy użyciu modelu segmentacyjnego
def analyze_image_segmentation(test_img_array, seg_model, ref_img_array):
    input_pair = np.concatenate([ref_img_array, test_img_array], axis=-1)
    input_pair = np.expand_dims(input_pair, axis=0)
    pred_mask = seg_model.predict(input_pair)[0,...,0]
    pred_mask_bin = (pred_mask > 0.5).astype(np.uint8) * 255
    
    test_cv = (test_img_array * 255).astype(np.uint8)
    test_cv = cv2.cvtColor(test_cv, cv2.COLOR_RGB2BGR)
    mask_color = cv2.applyColorMap(pred_mask_bin, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(test_cv, 0.7, mask_color, 0.5, 0)
    return overlay, pred_mask_bin

##############################################
# Funkcja oceny jakości zdjęcia na podstawie maski różnic
def classify_image(mask_bin, threshold):
    # Oblicz stosunek białych pikseli (różnic) do całkowitej liczby pikseli
    white_pixels = np.sum(mask_bin) / 255.0
    total_pixels = TARGET_SIZE[0] * TARGET_SIZE[1]
    ratio = white_pixels / total_pixels
    # Jeśli różnice przekraczają próg, uznajemy zdjęcie za złe
    return "Złe" if ratio > threshold else "Dobre", ratio

##############################################
# Interfejs aplikacji Streamlit
st.title("Detekcja różnic na PCB przy użyciu U-Net")
st.markdown("Aplikacja umożliwia trenowanie modelu segmentacyjnego na podstawie par zdjęć (referencyjnych i defektowych) oraz analizę nowego zdjęcia PCB. Wyniki prezentowane są dwiema metodami: maska generowana przez sieć (U-Net). Dodatkowo, na podstawie maski różnic określana jest jakość zdjęcia.")

# Sekcja przesyłania zdjęć treningowych
st.markdown("### Przesyłanie zdjęć treningowych")
tab_good, tab_bad = st.tabs(["Dobre zdjęcia (referencyjne)", "Złe zdjęcia (defekty)"])
with tab_good:
    uploaded_good = st.file_uploader("Prześlij dobre zdjęcie", type=["png", "jpg", "jpeg", "bmp"], key="good")
    if uploaded_good is not None:
        ref_saved = save_uploaded_file(uploaded_good, GOOD_DIR)
        st.image(ref_saved, caption="Dodane dobre zdjęcie", use_column_width=True)
with tab_bad:
    uploaded_bad = st.file_uploader("Prześlij złe zdjęcie", type=["png", "jpg", "jpeg", "bmp"], key="bad")
    if uploaded_bad is not None:
        bad_saved = save_uploaded_file(uploaded_bad, BAD_DIR)
        st.image(bad_saved, caption="Dodane złe zdjęcie", use_column_width=True)

st.markdown("---")
st.header("Trenowanie modelu segmentacyjnego")
if st.button("Rozpocznij trenowanie modelu segmentacyjnego"):
    tf.keras.backend.clear_session()  # Czyszczenie sesji przed trenowaniem
    train_segmentation_model()

st.markdown("---")
st.header("Analiza nowego zdjęcia PCB")
uploaded_test = st.file_uploader("Prześlij zdjęcie PCB do analizy", type=["png", "jpg", "jpeg", "bmp"], key="test")
if uploaded_test is not None:
    test_img = load_img(uploaded_test, target_size=TARGET_SIZE)
    test_img_array = img_to_array(test_img) / 255.0
    st.image(test_img, caption="Przesłane zdjęcie PCB", use_column_width=True)
    
    ref_path, ref_img_array = get_reference_image(TARGET_SIZE)
    if ref_img_array is None:
        st.error("Brak zdjęć referencyjnych w folderze: " + GOOD_DIR)
    else:
        st.image(ref_img_array, caption="Zdjęcie referencyjne", use_column_width=True)
        
        if os.path.exists(SEG_MODEL_PATH):
            tf.keras.backend.clear_session()  # Czyszczenie sesji przed wczytaniem modelu
            seg_model = load_model(SEG_MODEL_PATH)
        else:
            st.error("Model segmentacyjny nie został wytrenowany.")
        
        # Analiza przy użyciu modelu U-Net
        overlay_seg, mask_seg = analyze_image_segmentation(test_img_array, seg_model, ref_img_array)
        st.image(cv2.cvtColor(overlay_seg, cv2.COLOR_BGR2RGB),
                caption="Zdjęcie testowe z maską różnic (U-Net)",
                use_column_width=True)
        
        # Klasyfikacja zdjęcia na podstawie maski różnic (U-Net)
        classification, diff_ratio = classify_image(mask_seg, threshold=0.002)
        st.write(f"Klasyfikacja zdjęcia: **{classification}** (obszar różnic: {diff_ratio*100:.2f}%)")
else:
    st.error("Prześlij zdjęcie do analizy.")
