import streamlit as st
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from PIL import Image

loaded_iris = joblib.load('iris_pipe.pkl')



lunghezza_sepalo = st.slider("lunghezza del sepalo", 0, 1000, 25)
larghezza_sepalo = st.slider("larghezza del sepalo'", 0, 1000, 25)
lunghezza_petalo = st.slider("lunghezza del petalo", 0, 1000, 25)
larghezza_petalo = st.slider("larghezza del petalo", 0, 1000, 25)



fiore = {'lunghezza del sepalo':[lunghezza_sepalo],
        'larghezza del sepalo':[larghezza_sepalo],
        'lunghezza del petalo':[lunghezza_petalo],
        'larghezza del petalo': [larghezza_petalo],
        }


df_fiore = pd.DataFrame(fiore)

def che_fiore_e(df_fiore):
    
    pred = loaded_iris.predict(df_fiore)[0]
    if pred == 'Iris-setosa':
        st.image(Image.open("Iris_setosa.jpg"), use_container_width=True, caption= "Iris setosa"),
    if pred == 'Iris-versicolor':
        st.image(Image.open("Iris_versicolor.jpg"),use_container_width=True, caption= "Iris versicolor"),
    else:
        st.image(Image.open("iris_virginica.jpg"),use_container_width=True ,caption= "iris virginica")
                
        

    return pred

def main():
    
    st.text(f"il tuo fiore potrebbe essere: {che_fiore_e(df_fiore)}")
    
if __name__ == "__main__":
    main()    
    

