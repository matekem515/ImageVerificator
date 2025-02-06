# Detekcja różnic na PCB przy użyciu U-Net oraz klasycznego podejścia (Canny)

Projekt to aplikacja webowa oparta na [Streamlit](https://streamlit.io/), która umożliwia automatyczne wykrywanie defektów na płytkach drukowanych (PCB) poprzez porównanie obrazu testowego z referencyjnym zdjęciem dobrej płytki. Rozwiązanie łączy podejście oparte na uczeniu maszynowym (model segmentacyjny U-Net) .

## Funkcjonalności

- **Przesyłanie zdjęć treningowych:**  
  Użytkownik może przesyłać zdjęcia dobre (referencyjne) oraz złe (defektowe) do folderów `train_images/good` oraz `train_images/bad`.

- **Trenowanie modelu segmentacyjnego (U-Net):**  
  Model U-Net jest trenowany na danych treningowych, które są generowane automatycznie przez porównanie referencyjnego obrazu z defektowym zdjęciem. Ground truth maski są tworzone na podstawie różnicy między obrazami (po konwersji do odcieni szarości, progowaniu i 
  operacjach morfologicznych).

- **Analiza nowego zdjęcia PCB:**  
  Po przesłaniu nowego zdjęcia PCB aplikacja:
  - Wybiera referencyjne zdjęcie z folderu `train_images/good`.
  - Przygotowuje wejście dla modelu poprzez sklejenie referencyjnego i testowego obrazu (po 3 kanały każdy, co daje 6 kanałów).
  - Generuje maskę różnic przy użyciu wytrenowanego modelu U-Net.
  - Równolegle wykonuje klasyczne wykrywanie różnic metodą Canny.
  - Nakłada maskę na zdjęcie testowe, wizualizując defekty.
  - Oblicza procentową powierzchnię wykrytych różnic i klasyfikuje zdjęcie jako **"Dobre"** lub **"Złe"**.

## Wymagania

- Python 3.7+
- TensorFlow 2.x
- OpenCV (opencv-python)
- Streamlit
- NumPy

## Instalacja

1. **Utwórz wirtualne środowisko (opcjonalnie):**
   
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
2. **Zainstaluj wymagane biblioteki:**
   
   pip install tensorflow opencv-python streamlit numpy

## Uruchomienie aplikacji
1. **Skopiuj cały kod aplikacji do pliku app.py.**
2. **Uruchom aplikację za pomocą polecenia:**
    streamlit run app.py

## Instrukcja użycia

## Przesyłanie zdjęć treningowych
- Dobre zdjęcia (referencyjne):
- Prześlij zdjęcia dobrej płytki PCB. Zdjęcia te będą używane jako referencyjne do generowania danych treningowych.

## Złe zdjęcia (defekty):
- Prześlij zdjęcia PCB z defektami (np. brakujące elementy). Dane te zostaną sparowane z referencyjnymi i użyte do trenowania modelu.

## Trenowanie modelu segmentacyjnego

- Kliknij przycisk "Rozpocznij trenowanie modelu segmentacyjnego".
- Model U-Net zostanie wytrenowany na wygenerowanych parach zdjęć oraz ground truth maskach. Parametry treningu (np. liczba epok, validation_split, batch_size) możesz dostosować w funkcji model.fit().
  
## Analiza nowego zdjęcia PCB
- Prześlij nowe zdjęcie PCB do analizy w odpowiedniej sekcji.
- Aplikacja automatycznie pobierze referencyjne zdjęcie z folderu train_images/good i przygotuje dane wejściowe dla modelu.
- Model wygeneruje maskę różnic, która zostanie nałożona na zdjęcie testowe.
- Dodatkowo, klasyczne wykrywanie różnic metodą Canny zostanie wykonane.
- Na podstawie procentowego udziału wykrytych różnic, zdjęcie zostanie zaklasyfikowane jako "Dobre" lub "Złe".
  
## Dostosowanie treningu
W funkcji model.fit() możesz zmieniać następujące parametry:
- epochs – liczba epok treningowych (np. 100, 50 itp.).
- validation_split – procent danych przeznaczonych na walidację (np. 0.2).
- batch_size – rozmiar partii (np. 32).
Możesz również dodać mechanizmy takie jak EarlyStopping czy ReduceLROnPlateau za pomocą callbacks, aby zoptymalizować proces trenowania.

## Uwagi
- Ground truth maski: Automatyczne generowanie masek różnic (na podstawie różnicy między referencyjnym a defektowym zdjęciem) może wymagać korekty w zależności od warunków oświetleniowych i jakości zdjęć.
- Clear Session: W kodzie zastosowano tf.keras.backend.clear_session() przed trenowaniem oraz przed wczytaniem modelu, aby uniknąć konfliktów wynikających z poprzednich instancji modeli.
- Jakość danych: Wynik detekcji defektów zależy od jakości i reprezentatywności danych treningowych. Zaleca się zebranie różnorodnych zdjęć oraz, jeśli to możliwe, ręczną korektę ground truth masek.
  
## Podsumowanie
Aplikacja łączy podejście oparte na sieciach neuronowych (U-Net) z klasycznymi metodami wykrywania różnic (Canny) w celu automatycznego wykrywania defektów na PCB. Użytkownik ma możliwość:
- Przesyłania zdjęć treningowych i analizy nowego zdjęcia PCB.
- Trenowania modelu segmentacyjnego.
- Wizualizacji wyników analizy – maski generowanej przez U-Net oraz zarysu różnic metodą Canny.
- Klasyfikacji zdjęcia jako "Dobre" lub "Złe" na podstawie procentowego udziału wykrytych różnic.
