import csv
import speech_recognition as sr
import random
import time

# Load the English and Hindi silly words from the CSV file
word_pairs = []
with open('english_hindi_silly_words.csv', mode='r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        word_pairs.append((row[0], row[1]))

# Initialize recognizer
recognizer = sr.Recognizer()

def get_random_word():
    """Choose a random word pair (English, Hindi) from the list."""
    return random.choice(word_pairs)

def listen_for_word():
    """Listen to the microphone and recognize speech."""
    with sr.Microphone() as source:
        # Adjust recognizer sensitivity to ambient noise
        recognizer.adjust_for_ambient_noise(source, duration=1.0)  # Adjust duration for noise
        print("Listening...")

        try:
            # Listen for a word, with enough time to account for pauses
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)  # Increased timeout for more flexibility
            print("Recognizing...")

            # Use Google speech recognition with a quick response
            user_word = recognizer.recognize_google(audio, show_all=False).lower()
            return user_word

        except sr.WaitTimeoutError:
            print("Timeout: Please speak a bit louder or closer.")
            return None
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand the audio. Please try again.") 
            return None
        except sr.RequestError as e:
            print(f"There was an issue with the speech recognition service: {e}")
            return None

def check_word_accuracy(user_word, english_word, hindi_word):
    """Check if the user input matches either the English or Hindi word."""
    # Normalize the comparison by stripping spaces and converting to lowercase
    user_word = user_word.strip().lower()
    english_word = english_word.strip().lower()
    hindi_word = hindi_word.strip().lower()

    if user_word == english_word or user_word == hindi_word:
        print("Correct!")
        return True
    else:
        print(f"Incorrect! The correct word was: {english_word} or {hindi_word}")
        return False

def main():
    print("Speech Recognition Word Test")
    while True:
        # Get a random word from the list
        english_word, hindi_word = get_random_word()
        print(f"Your word is: {english_word} (Hindi translation: {hindi_word})")
        print("Please repeat the word after the beep.")

        # Simulate a beep before user starts speaking
        time.sleep(1)

        # Listen to the user's speech
        user_word = None
        user_word = listen_for_word()

        if user_word:
            # Check if the user repeated the word correctly
            if check_word_accuracy(user_word, english_word, hindi_word):
                print("You got it right!")
            else:
                print("Try again.")
        
        print("\nNext round...")
        time.sleep(2)  # Pause before the next round

if __name__ == "__main__":
    main()
