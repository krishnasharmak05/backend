from turtle import up
import flask
import requests
import bs4

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

app = flask.Flask(__name__)


def upload_song_and_save_midi():
    

# Setup WebDriver (you can replace with the path to your browser's driver)
    driver = webdriver.Chrome()

# Open the website
    driver.get('https://www.musictomidi.com/')  # Replace with the URL of the website you need

# Locate the file input element
    file_input = driver.find_element(By.XPATH, '//*[@type="file"]')
    print(file_input)

# Upload the file (provide the full file path)
    file_input.send_keys('C:/Users/krish/Downloads/videoplayback.mp3')  # Replace with the full path of the file you need to upload

# Wait for a few seconds to allow the upload process to complete
    time.sleep(15)

# Optionally, close the browser after the task is complete
    driver.quit()

    return




if __name__ == "__main__":
    upload_song_and_save_midi()