import streamlit as st
import tempfile
import os
import pandas as pd
import time
import speech_recognition as sr
import pyttsx3 
from COMPUTER_Vision import dimension_reducing, object_detection, propmpt, image_size 
from bardapi import BardCookies

cookie_dict = {
    "__Secure-1PSID": "dwgMTws4XcmYSqvbU7CQVmISNZam9Zl40vaq_WXWPHKXsB1bfNaIlaYM6xiBNaO_0hyKVw.",
    "__Secure-1PSIDTS": "sidts-CjEBPVxjSvat-F5J7RbmGymQwu2rGosgfCzur3bzXdGXzETxCBTzY1n9tKJ75ETbuqrTEAA",
    "__Secure-1PSIDCC": "ABTWhQGvdcB0RBFQY8DDjgY0o6BBFyLHRahRQ9AU2E5e5Dm9S7L1szZMfbCLBky1s4wk-y-me_L8",
}

bard = BardCookies(cookie_dict=cookie_dict)

# Set the flag to indicate whether the initial explanation has been given


prompt_1 = '''Objectif : Développer un modèle performant pour guider les personnes aveugles en utilisant des lunettes équipées de caméras capables de scanner l'environnement. Le modèle doit détecter les objets grâce à un modèle de détection d'objets et estimer la distance de ces objets à l'aide d'un modèle d'estimation de profondeur. Les données de détection d'objets et de profondeur seront fournies au modèle de langage par le biais d'une question ou d'une demande de la personne. Les lunettes sont également équipées de modèles de conversion de la parole en texte et vice versa pour faciliter une interaction auditive. Le modèle doit être capable d'analyser les données d'entrée, fournir des réponses claires et rapides, et remplacer virtuellement les yeux de la personne, en informant sur les éléments de l'environnement ou en répondant à des questions. Les matrices de profondeur fournissent des informations sur la distance de chaque pixel dans l'image. La détection des objets fournit des informations détaillées sur les objets présents, leur type, leur position et leur profondeur. Pour répondre à des questions comme la présence d'obstacles à gauche, une matrice de profondeur réduite est utilisée en effectuant une moyenne sur des blocs pour simplifier l'analyse. Ces informations seront utilisées pour répondre aux demandes spécifiques de la personne et maintenir une conversation fluide, même si elle n'est pas directement liée à son environnement immédiat.

Prompt :
Dimension de l’image : (X*Y) - Exemple : (1280*960)
Matrice de profondeur réduite : La dimension initiale de la matrice était égale à la dimension de l’image, donc chaque coefficient dans la matrice de profondeur représente la profondeur du pixel dans l’image. Après réduction, on obtient 64 régions.
Analyse nécessaire : La région x=1, y=1 représente la moyenne des pixels entre x1=0 et x2=X/8 et y=0 et y=Y/8. En général, la région x=k et y=s représente la moyenne des pixels entre x1=(k-1)*X/8 et x2=k*X/8 et y=(k-1)*Y/8 et y=k*Y/8. Exemple : [[2.92, 3.51, 3.48, 3.75, 3.37, 2.97, 3.02, 3.31], [1.87, 3.20, 4.07, 4.59, 4.26, 3.94, 3.68, 3.68], [1.77, 3.75, 6.03, 6.51, 6.07, 6.09, 6.65, 4.28], [1.72, 3.99, 6.52, 5.86, 5.83, 5.38, 7.93, 4.37], [1.72, 2.57, 4.04, 4.85, 4.86, 4.83, 4.55, 3.45], [1.78, 2.72, 4.05, 3.99, 3.71, 3.44, 3.50, 3.32], [1.87, 2.64, 3.38, 3.32, 3.21, 3.21, 3.25, 3.16], [1.94, 2.46, 2.57, 2.54, 2.52, 2.51, 2.51, 2.53]]

Objets détectés : Dans cette partie, tu trouveras le nom des différents objets détectés avec leur position relative dans l’image de forme [xmin, ymin, xmax, ymax] et la certitude de la détection qui est le pourcentage de confiance.
Analyse nécessaire : Plus la confiance est proche de 1, plus tu es sûr de l’existence de l’objet détecté dans la région détectée. Plus la confiance est proche de 0.5, plus tu n’es pas sûr de l’existence de l’objet détecté dans la région détectée. La position de l’objet se calcule par le moyennage suivant : position = ((xmin+xmax)/2, (ymin+ymax)/2). Pour savoir la position relative de l’objet, divise la position sur la dimension de l’image : position relative((xmin+xmax)/(2*X), (ymin+ymax)/(2*Y)). Si (xmin+xmax)/(2*X) < 0,5, l’objet est à gauche. Si (xmin+xmax)/(2*X) > 0,5, l’objet est à droite. Si (xmin+xmax)/(2*X) est proche de 0,5, le projet est au milieu. Si (ymin+ymax)/(2*X) < 0,5, l’objet est en haut. Si (xmin+xmax)/(2*X) > 0,5, l’objet est en bas. Si (xmin+xmax)/(2*X) est proche de 0,5, le projet est au milieu. (La question du en haut et en bas n’est pas importante si la profondeur > 5). La profondeur montre le degré de distance de l’objet des lunettes. Si la profondeur < 2, l’objet est proche. Si 2 < la profondeur < 8, l’objet est au milieu. Si 8 < la profondeur, l’objet est loin.

Exemples : "Chair" à la position [256.59, 485.54, 339.68, 534.69] avec une confiance de 0.657 et une profondeur de 4.28. "Book" à différentes positions avec confiance et profondeur. "Clock" à diverses positions avec confiance et profondeur. "Dining table" avec confiance et profondeur.

Demande : "Est-ce qu'il y a des personnes à côté de moi ?"
Completion : "Non, il n'y a personne. Cependant, sois attentif à ta gauche, car la distance dans la première région de la matrice de profondeur indique une proximité entre 1.7 et 2.9." Assiste la personne aveugle (si c'est compris, réponds par "on commence").
'''

r = sr.Recognizer()

def SpeakText(command):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    for voice in voices:
        if "english" in voice.name.lower():
            engine.setProperty('voice', voice.id)
            break
    engine.setProperty('rate', 150) 
    engine.setProperty('volume', 1.0)
    engine.say(command)
    engine.runAndWait()
def main():
    st.title("Eldians")
    initial_explanation_given = False
    # Add file uploader for images
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        # Save the uploaded image to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())
        image_path = temp_file.name

        S = image_size(image_path)
        A1, B = object_detection(image_path)
        C = dimension_reducing(B)

        if not initial_explanation_given:
            # If the initial explanation has not been given, provide it and set the flag to True
            response = bard.get_answer(prompt_1)['content']
            initial_explanation_given = True
            st.text("Ready")
        # Button to start the conversation
        if st.button("Start Conversation"):
            with sr.Microphone() as source2:
                r.adjust_for_ambient_noise(source2, duration=0.2)
                st.text("Listening for speech input...")

                # Create a button to stop the conversation
                stop_button = st.button("Stop Conversation")

                # Record audio
                audio2 = r.listen(source2)
                MyText = r.recognize_google(audio2)
                MyText = MyText.lower()

                # Display vocal prompt as text on the interface
                st.text(f"Vocal Prompt: {MyText}")

                while not stop_button:
                    # Perform the necessary operations with the speech input
                    response = bard.get_answer(str(propmpt(S, A1, C, MyText)))['content']
                    SpeakText(response)
                    # Remplacer les astérisques par une chaîne vide
                    response = response.replace('*', '')
                    # Print the response
                    st.text(f"Response: {response}")

                    # Record audio for the next iteration
                    audio2 = r.listen(source2)
                    MyText = r.recognize_google(audio2)
                    MyText = MyText.lower()

                    # Display vocal prompt as text on the interface
                    st.text(f"Vocal Prompt: {MyText}")

        # Close and delete the temporary file
        temp_file.close()
        os.unlink(image_path)

 
if __name__ == "__main__":
    main()
