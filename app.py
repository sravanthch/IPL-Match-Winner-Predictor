import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="IPL Match Predictor",
    page_icon="🏏",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS for Premium Design & Mobile Responsiveness
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    :root {
        --primary: #6C5CE7;
        --secondary: #A29BFE;
        --bg: #0F172A;
        --card-bg: rgba(30, 41, 59, 0.7);
        --text: #F8FAFC;
    }

    * {
        font-family: 'Outfit', sans-serif !important;
    }

    /* Main Background */
    .stApp {
        background: radial-gradient(circle at top right, #1E1B4B, #0F172A);
        color: var(--text);
    }


    /* Header styling */
    h1 {
        background: linear-gradient(90deg, #818CF8, #C084FC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
        text-align: center;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    .subheader {
        text-align: center;
        color: #94A3B8;
        margin-bottom: 2rem;
    }

    /* Button Styling */
    .stButton > button {
        background: linear-gradient(90deg, #6366F1, #A855F7) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6) !important;
    }

    /* Selectbox styling */
    .stSelectbox label {
        color: #CBD5E1 !important;
        font-weight: 500 !important;
    }
    
    /* Result styling */
    .prediction-card {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid #10B981;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 2rem;
    }
    
    .predicted-winner {
        font-size: 1.8rem;
        font-weight: 700;
        color: #10B981;
        margin-top: 0.5rem;
    }

    /* Hide Streamlit elements completely */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDecoration {display:none;}
    
    /* Remove padding at the top */
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
    }

    
    </style>
""", unsafe_allow_html=True)

# Load the model and encoders
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('model.pkl')
        encoders = joblib.load('encoders.pkl')
        return model, encoders
    except FileNotFoundError:
        return None, None

model, encoders = load_artifacts()

if model is None or encoders is None:
    st.warning("Model artifacts not found. Please run `train_model.py` first.")
    st.stop()

# Header Section
st.markdown("<h1>🏏 IPL 2026 Match Winner</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Next-Gen Winning Probability Predictor</p>", unsafe_allow_html=True)

# Input Form in a Card
with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    
    # Helper function to get sorted classes
    def get_classes(encoder):
        return sorted(list(encoder.classes_))

    team_list = get_classes(encoders['team1'])
    venue_list = get_classes(encoders['venue'])
    toss_decision_list = get_classes(encoders['toss_decision'])

    # UI Layout
    col1, col2 = st.columns(2)

    with col1:
        team1 = st.selectbox("Team 1 (Batting)", team_list)
    with col2:
        team2 = st.selectbox("Team 2 (Bowling)", team_list, index=1 if len(team_list) > 1 else 0)

    venue = st.selectbox("🏟️ Venue", venue_list)

    col3, col4 = st.columns(2)
    with col3:
        toss_winner = st.selectbox("🪙 Toss Winner", team_list)
    with col4:
        toss_decision = st.selectbox("📋 Toss Decision", toss_decision_list)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Calculate Probability 🔮", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

if predict_btn:
    if team1 == team2:
        st.error("Team 1 and Team 2 cannot be the same!")
    else:
        # Prepare the input data
        input_data = {
            'team1': team1,
            'team2': team2,
            'venue': venue,
            'toss_winner': toss_winner,
            'toss_decision': toss_decision
        }
        
        try:
            # Encode inputs
            encoded_input = []
            features = ['team1', 'team2', 'venue', 'toss_winner', 'toss_decision']
            
            for f in features:
                val = input_data[f]
                encoded_val = encoders[f].transform([val])[0]
                encoded_input.append(encoded_val)
                
            # Predict
            prediction_encoded = model.predict([encoded_input])[0]
            predicted_winner = encoders['target'].inverse_transform([prediction_encoded])[0]
            
            # Predict Probabilities
            probs = model.predict_proba([encoded_input])[0]
            prob_team1 = 0
            prob_team2 = 0
            
            # Get index in the target encoder classes list
            target_classes = list(encoders['target'].classes_)
            
            try:
                t1_idx = target_classes.index(team1)
                prob_team1 = probs[t1_idx]
            except ValueError:
                pass
                
            try:
                t2_idx = target_classes.index(team2)
                prob_team2 = probs[t2_idx]
            except ValueError:
                pass
            
            # Result Card
            st.markdown(f"""
                <div class="prediction-card">
                    <div style="color: #94A3B8; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;">Predicted Winner</div>
                    <div class="predicted-winner">🏆 {predicted_winner}</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Display stats
            st.markdown("### 📊 Win Probability")
            if prob_team1 > 0 or prob_team2 > 0:
                total_prob = prob_team1 + prob_team2
                if total_prob > 0:
                    p1 = (prob_team1 / total_prob)
                    p2 = (prob_team2 / total_prob)
                    
                    st.markdown(f"**{team1}**: {p1*100:.1f}%")
                    st.progress(float(p1))
                    
                    st.markdown(f"**{team2}**: {p2*100:.1f}%")
                    st.progress(float(p2))
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: #64748B; font-size: 0.8rem; margin-bottom: 2rem;'>Developed by Sravanth Chittanuru<br><br></div>", unsafe_allow_html=True)
