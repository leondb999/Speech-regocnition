# SpeqQ - Audio Recognition DHBW Mannheim 6. Semester Project 

## About

* User records word from website
* Model makes prediction  status: working/prototype
* User sees the result in Percent


## Installation
Start Server from terminal

    uvicorn main:app --reload

Change absolute filepath in File **main.py (Zeile 268)** to your local path to the folder **audio_files**:

    path= r"C:/2019-Leon-eigene-Dateien/Studium/6-Semester/Integrationsseminar Speech-regocnition/audio_files/"



## Screenshots




## Problems
- Könnte sein dass wenn server das erste mal gestartet ist, dass man seite nochmal neuladen muss bevor es funktioniert
- ML Modell braucht einige Zeit um geladen zu werden
