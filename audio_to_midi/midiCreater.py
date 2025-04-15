import time
import os
from unittest import result
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import numpy as np
import pretty_midi
from scipy.spatial.distance import cosine, euclidean
from fastdtw import fastdtw
from collections import Counter
from difflib import SequenceMatcher
from scipy.fft import fft
from scipy.stats import entropy

from fastapi import FastAPI

app = FastAPI()

MIDI_FOLDER = os.path.abspath("./output")

def create_midi(mp3_file):
    os.makedirs(MIDI_FOLDER, exist_ok=True)
    options = webdriver.ChromeOptions()
    prefs = {"download.default_directory": MIDI_FOLDER}
    options.add_experimental_option("prefs", prefs)
    options.add_argument("--headless")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=options
    )
    try:
        driver.get("https://www.musictomidi.com/")
        file_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//input[@type='file']"))
        )

        file_path = os.path.abspath(f"./inputs/{mp3_file}.mp3")
        file_input.send_keys(file_path)
        
        download_button = WebDriverWait(driver, 300).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Download MIDI')]"))
        )
        driver.execute_script("arguments[0].click();", download_button)
        print("Download button clicked!")

        time.sleep(5) 
        midi_file = None
        for _ in range(30):
            files = [f for f in os.listdir(MIDI_FOLDER) if f.endswith(".mid")]
            if files:
                midi_file = files[0]
                break
            time.sleep(1)

        if midi_file:
            print(f"MIDI file downloaded: {midi_file}")
        else:
            print("MIDI file download failed.")

    except Exception as e:
        print(f"An error occurred:", str(e))
        print("Exiting in 5....")
        time.sleep(5)

    finally:
        driver.quit() 


def extract_midi_features(midi_file):
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    pitches, note_durations, intervals_between_notes = [], [], []
    
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            prev_start = 0
            for note in instrument.notes:
                pitches.append(note.pitch)
                note_durations.append(note.end - note.start)  
                intervals_between_notes.append(note.start - prev_start)
                prev_start = note.start

    return np.array(pitches), np.array(note_durations), np.array(intervals_between_notes)


def cosine_similarity(seq1, seq2):
    min_len = min(len(seq1), len(seq2))
    seq1, seq2 = seq1[:min_len], seq2[:min_len]
    return 1 - cosine(seq1, seq2) if len(seq1) > 0 and len(seq2) > 0 else 0

# melody comparison
def dynamic_time_warping_similarity(seq1, seq2):
    distance, _ = fastdtw(seq1.reshape(-1, 1), seq2.reshape(-1, 1), dist=euclidean)
    return 1 / (1 + distance)

# Compare note duration sequences
def rhythm_similarity(seq1, seq2):
    return cosine_similarity(seq1, seq2)

# Compare chord distributions
def harmonic_similarity(seq1, seq2):
    hist1, hist2 = Counter(seq1), Counter(seq2)
    all_chords = set(hist1.keys()).union(set(hist2.keys()))
    
    vec1 = np.array([hist1[n] for n in all_chords])
    vec2 = np.array([hist2[n] for n in all_chords])
    
    return 1 - cosine(vec1, vec2) if len(vec1) > 0 and len(vec2) > 0 else 0

# Compare sequences of 3 consecutive notes
def ngrams_fingerprinting(seq1, seq2, n=3):
    grams1 = set(tuple(seq1[i:i+n]) for i in range(len(seq1)-n+1))
    grams2 = set(tuple(seq2[i:i+n]) for i in range(len(seq2)-n+1))
    intersection = len(grams1.intersection(grams2))
    union = len(grams1.union(grams2))
    return intersection / union if union != 0 else 0

# Fourier Transform Similarity - Compare frequency components
def fourier_similarity(seq1, seq2):
    fft1, fft2 = fft(seq1), fft(seq2)
    min_len = min(len(fft1), len(fft2))
    return cosine_similarity(np.abs(fft1[:min_len]), np.abs(fft2[:min_len]))

# Measures musical complexity
def entropy_similarity(seq1, seq2):
    return 1 - abs(entropy(np.bincount(seq1)) - entropy(np.bincount(seq2)))

# Function to compare one MIDI file against a list
def compare_midi_files(main_midi, midi_list):
    main_pitches, main_durations, main_intervals = extract_midi_features(os.path.abspath(os.path.join("output", main_midi)))
    similarities = {}

    for midi_file in midi_list:
        other_pitches, other_durations, other_intervals = extract_midi_features(midi_file)
        
        cosine_sim = cosine_similarity(main_pitches, other_pitches)
        dtw_sim = dynamic_time_warping_similarity(main_pitches, other_pitches)
        rhythm_sim = rhythm_similarity(main_durations, other_durations)
        harmonic_sim = harmonic_similarity(main_pitches, other_pitches)
        ngram_sim = ngrams_fingerprinting(main_pitches, other_pitches)
        fourier_sim = fourier_similarity(main_pitches, other_pitches)
        entropy_sim = entropy_similarity(main_pitches, other_pitches)

        similarities[midi_file] = {
            "Cosine Similarity": round(cosine_sim, 3),
            "DTW Similarity": round(dtw_sim, 3),
            "Rhythm Similarity": round(rhythm_sim, 3),
            "Harmonic Similarity": round(harmonic_sim, 3),
            "n-gram Similarity": round(ngram_sim, 3),
            "Fourier Similarity": round(fourier_sim, 3),
            "Entropy Similarity": round(entropy_sim, 3),
        }
    
    return similarities









# ADDING a nn

# Feature extraction

# import pretty_midi
# import numpy as np
# # import tensorflow as tf
# from sklearn.preprocessing import MinMaxScaler
# from scipy.fft import fft
# from collections import Counter

# # Extract features from a MIDI file
# def extract_midi_features(midi_file):
#     midi_data = pretty_midi.PrettyMIDI(midi_file)
    
#     pitches, durations, intervals = [], [], []
#     for instrument in midi_data.instruments:
#         if not instrument.is_drum:
#             prev_start = 0
#             for note in instrument.notes:
#                 pitches.append(note.pitch)
#                 durations.append(note.end - note.start)
#                 intervals.append(note.start - prev_start)
#                 prev_start = note.start

#     # Convert to numpy arrays
#     pitches, durations, intervals = np.array(pitches), np.array(durations), np.array(intervals)

#     # Normalize features
#     scaler = MinMaxScaler()
#     if len(pitches) > 0:
#         pitches = scaler.fit_transform(pitches.reshape(-1, 1)).flatten()
#         durations = scaler.fit_transform(durations.reshape(-1, 1)).flatten()
#         intervals = scaler.fit_transform(intervals.reshape(-1, 1)).flatten()
    
#     # Compute Fourier Transform of the pitch sequence
#     if len(pitches) > 0:
#         fft_features = np.abs(fft(pitches))
#         fft_features = scaler.fit_transform(fft_features.reshape(-1, 1)).flatten()
#     else:
#         fft_features = np.zeros(10)

#     # Combine features
#     feature_vector = np.concatenate([pitches[:50], durations[:50], intervals[:50], fft_features[:10]])  # Truncate or pad
    
#     return feature_vector


# # Step 2: Preparing training data (We'll assume we have pairs of MIDI files labeled as "similar" (1) or "different" (0) to train our neural network.)
# import os
# import random
# import numpy as np

# # Load MIDI files and create dataset
# def create_dataset(midi_folder, num_samples=1000):
#     midi_files = [os.path.join(midi_folder, f) for f in os.listdir(midi_folder) if f.endswith(".mid")]
#     dataset = []

#     for _ in range(num_samples):
#         midi1, midi2 = random.sample(midi_files, 2)
#         features1, features2 = extract_midi_features(midi1), extract_midi_features(midi2)

#         # Compute similarity (DTW, cosine, etc.)
#         similarity_score = np.random.choice([0, 1])  # Assume labels (0 = different, 1 = similar)

#         dataset.append((features1, features2, similarity_score))

#     return dataset

# # Convert to numpy arrays
# def prepare_training_data(dataset):
#     X1, X2, Y = [], [], []
#     for f1, f2, label in dataset:
#         X1.append(f1)
#         X2.append(f2)
#         Y.append(label)

#     return np.array(X1), np.array(X2), np.array(Y)

# Load dataset
# dataset = create_dataset("output/")
# print(midi_folder)
# X1_train, X2_train, Y_train = prepare_training_data(dataset)



# Step 4: Build the nn

# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, concatenate
# from tensorflow.keras.optimizers import Adam

# # Define the neural network for feature extraction
# def create_feature_extractor(input_shape):
#     input_layer = Input(shape=(input_shape,))
#     x = Dense(128, activation='relu')(input_layer)
#     x = Dropout(0.3)(x)
#     x = Dense(64, activation='relu')(x)
#     x = Dropout(0.3)(x)
#     x = Dense(32, activation='relu')(x)
#     return Model(inputs=input_layer, outputs=x)

# # Define Siamese Network
# input_shape = X1_train.shape[1]

# feature_extractor = create_feature_extractor(input_shape)

# input1 = Input(shape=(input_shape,))
# input2 = Input(shape=(input_shape,))

# encoded1 = feature_extractor(input1)
# encoded2 = feature_extractor(input2)

# merged = concatenate([encoded1, encoded2])
# x = Dense(32, activation='relu')(merged)
# x = Dense(16, activation='relu')(x)
# output = Dense(1, activation='sigmoid')(x)  # Output similarity score

# model = Model(inputs=[input1, input2], outputs=output)
# model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# model.summary()


# # Step 5: Train the model
# model.fit([X1_train, X2_train], Y_train, epochs=20, batch_size=32, validation_split=0.2)

# # Step 6: Plagiarism detection:
# def predict_similarity(midi1, midi2, model):
#     features1, features2 = extract_midi_features(midi1), extract_midi_features(midi2)
#     similarity_score = model.predict([np.array([features1]), np.array([features2])])[0][0]
#     return similarity_score

# # Example: Compare two MIDI files
# midi_file1 = "output/input.mid"
# midi_file2 = "output/input.mid"
# similarity = predict_similarity(midi_file1, midi_file2, model)
# print(f"Predicted Similarity Score: {similarity}")




@app.get("/process_midi")
async def main():
    main_midi_file = None
    midi_folder = "output/"
    
    for file in os.listdir("inputs"):
        if file.split(".")[0]+".mid" not in os.listdir("output"):
            main_midi_file = file.split(".")[0]+".mid"
            create_midi(file.split(".")[0])
            midi_files = [os.path.join(midi_folder, f) for f in os.listdir(midi_folder) if f.endswith(".mid")]

            results = compare_midi_files(main_midi_file, midi_files)

            sorted_results = sorted(results.items(), key=lambda x: x[1]["DTW Similarity"], reverse=True)
            for midi, scores in sorted_results:
                print(f"{midi}: {scores}")
            result_dict = {midi: scores for midi, scores in sorted_results}
            return result_dict
    return {"message": "No new MIDI files to process."}


if __name__ == "__main__":
    main()
