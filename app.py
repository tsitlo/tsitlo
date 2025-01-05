{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "0a2548a1-db5d-4c81-8345-ee5c22289672",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package stopwords to\n",
      "[nltk_data]     C:\\Users\\lizaa\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package stopwords is already up-to-date!\n",
      "C:\\Users\\lizaa\\anaconda3\\Lib\\site-packages\\sklearn\\base.py:380: InconsistentVersionWarning: Trying to unpickle estimator TfidfTransformer from version 1.0.2 when using version 1.6.0. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:\n",
      "https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations\n",
      "  warnings.warn(\n",
      "C:\\Users\\lizaa\\anaconda3\\Lib\\site-packages\\sklearn\\base.py:380: InconsistentVersionWarning: Trying to unpickle estimator TfidfVectorizer from version 1.0.2 when using version 1.6.0. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:\n",
      "https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations\n",
      "  warnings.warn(\n",
      "2025-01-05 18:59:08.617 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-05 18:59:09.227 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\lizaa\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2025-01-05 18:59:09.227 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-05 18:59:09.237 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-05 18:59:09.237 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-05 18:59:09.240 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-05 18:59:09.242 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-05 18:59:09.243 Session state does not function when running a script without `streamlit run`\n",
      "2025-01-05 18:59:09.244 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-05 18:59:09.246 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-05 18:59:09.248 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-05 18:59:09.248 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-05 18:59:09.250 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-05 18:59:09.251 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-05 18:59:09.253 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import joblib\n",
    "import re\n",
    "from nltk.stem import PorterStemmer\n",
    "from nltk.corpus import stopwords\n",
    "import nltk\n",
    "\n",
    "# Download stopwords if you haven't already\n",
    "nltk.download('stopwords')\n",
    "\n",
    "# Load the model and vectorizer\n",
    "model = joblib.load('Logistic_model')\n",
    "vectorizer = joblib.load('Vectorizer')\n",
    "\n",
    "def stemming(content):\n",
    "    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)\n",
    "    stemmed_content = stemmed_content.lower()\n",
    "    stemmed_content = stemmed_content.split()\n",
    "    port_stem = PorterStemmer()\n",
    "    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]\n",
    "    return ' '.join(stemmed_content)\n",
    "\n",
    "st.title(\"Article Category Predictor\")\n",
    "\n",
    "user_input = st.text_area(\"Enter Article Text:\")\n",
    "\n",
    "if st.button(\"Predict Category\"):\n",
    "    if user_input:\n",
    "        # Preprocess the user input\n",
    "        processed_input = stemming(user_input)\n",
    "        vectorized_input = vectorizer.transform([processed_input])\n",
    "        \n",
    "        # Make prediction\n",
    "        prediction = model.predict(vectorized_input)\n",
    "        st.write(f\"The predicted category is: {prediction[0]}\")\n",
    "    else:\n",
    "        st.write(\"Please enter some text to get a prediction.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "7c00e87f-8dd8-4085-a9a2-4e892dbc806b",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
