# 🏏 IPL 2026 Match Winner Predictor

A Machine Learning web application built with Python and Streamlit to predict the winning team of Indian Premier League (IPL) matches. 

The ML model is trained on historical ball-by-ball IPL match data, taking into account `Team 1`, `Team 2`, `Venue`, `Toss Winner`, and `Toss Decision` to establish head-to-head performance and make predictions for upcoming matches!

## 🚀 Live Demo Deployment
This app is designed to be easily deployed to Streamlit Community Cloud. 
Because the Machine Learning model is pre-trained and saved into `.pkl` files, **you do not need the large dataset to run the app!** Only the code files and the model artifacts need to be pushed to your repository.

## 🛠️ Local Installation & Setup

1. **Clone this repository**
   ```bash
   git clone <your-repo-url>
   cd "ML IPL Predictor"
   ```

2. **Install Dependencies**
   Install the necessary Python packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App**
   Launch our pre-trained model directly in your browser:
   ```bash
   python -m streamlit run app.py
   ```

---

## 🧠 Retraining the Model (Optional)

If you wish to update the model or train it on newer data, you will need the full dataset. 

1. **Download the Dataset**
   The training script requires the historical ball-by-ball IPL dataset, which is approximately `100MB` and thus **not** included in this repository.
   - Download it from [Kaggle](https://www.kaggle.com/).
   - Create a folder named `Dataset` in the root of this project.
   - Extract and place the dataset as `Dataset/IPL.csv`.

2. **Run the Training Script**
   ```bash
   python train_model.py
   ```
   This will process the dataset, convert ball-by-ball data into match-level statistics, train a Random Forest Classifier, and finally save fresh versions of `model.pkl` and `encoders.pkl` that your app uses.

## 📁 Repository Structure
- `app.py`: The Streamlit web interface.
- `train_model.py`: Script to preprocess data and train the AI model.
- `requirements.txt`: Python package dependencies.
- `model.pkl`: The saved Random Forest model (generated after training).
- `encoders.pkl`: Saved label encoders for translating text (Team names, Venues) to model inputs.
- `.gitignore`: Ensures large datasets and unnecessary cache files aren't accidentally pushed to GitHub.
