import streamlit as st
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

loaded_iris = joblib.load('iris_pipe.pkl')



lunghezza_sepalo = st.slider("lunghezza del sepalo", 0, 130, 25)
larghezza_sepalo = st.slider("larghezza del sepalo'", 0, 130, 25)
lunghezza_petalo = st.slider("lunghezza del petalo", 0, 130, 25)
larghezza_petalo = st.slider("larghezza del petalo", 0, 130, 25)



fiore = {'lunghezza del sepalo':[lunghezza_sepalo],
        'larghezza del sepalo':[larghezza_sepalo],
        'lunghezza del petalo':[lunghezza_petalo],
        'larghezza del petalo': [larghezza_petalo],
        }

df_fiore = pd.DataFrame(fiore)

def che_fiore_e(df_fiore):
    
    pred = loaded_iris.predict(df_fiore)[0]
    
    return pred

def main():
    
    st.text(f"il tuo fiore potrebbe essere: {che_fiore_e(df_fiore)}")
    
if __name__ == "__main__":
    main()    
    

