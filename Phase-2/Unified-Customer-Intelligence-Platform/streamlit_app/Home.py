import streamlit as st
from streamlit_app.utils.auth import signup, login, is_authenticated, get_current_user, logout

st.set_page_config(
    page_title="Unified Customer Intelligence Platform",
    layout="wide",
    page_icon="🎯"
)

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "user" not in st.session_state:
    st.session_state["user"] = None
if "show_login" not in st.session_state:
    st.session_state["show_login"] = False

# Authentication UI
if not is_authenticated():
    st.markdown('<p style="font-size: 3rem; font-weight: bold; color: #1f77b4; text-align: center;">🎯 Unified Customer Intelligence Platform</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 1.5rem; color: #666; text-align: center; margin-bottom: 2rem;">Please Login or Signup to Continue</p>', unsafe_allow_html=True)
    
    # Auto-select Login tab if signup was successful
    if st.session_state.get("show_login", False):
        default_tab = 0
        st.session_state["show_login"] = False
    else:
        default_tab = 0
    
    tab1, tab2 = st.tabs(["Login", "Signup"])
    
    with tab1:
        st.subheader("Login")
        login_email = st.text_input("Email", key="login_email")
        login_password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login"):
            success, user = login(login_email, login_password)
            if success:
                st.session_state["authenticated"] = True
                st.session_state["user"] = user
                st.success(f"Welcome back, {user['name']}!")
                st.rerun()
            else:
                st.error("Invalid email or password")
    
    with tab2:
        st.subheader("Signup")
        signup_name = st.text_input("Full Name", key="signup_name")
        signup_email = st.text_input("Email", key="signup_email")
        signup_phone = st.text_input("Phone Number (with country code, e.g., +1234567890)", key="signup_phone")
        signup_password = st.text_input("Password", type="password", key="signup_password")
        signup_confirm = st.text_input("Confirm Password", type="password", key="signup_confirm")
        
        if st.button("Signup"):
            if not all([signup_name, signup_email, signup_phone, signup_password]):
                st.error("All fields are required")
            elif signup_password != signup_confirm:
                st.error("Passwords do not match")
            elif len(signup_password) < 6:
                st.error("Password must be at least 6 characters")
            else:
                success, message = signup(signup_name, signup_email, signup_phone, signup_password)
                if success:
                    st.success(message + " Redirecting to login...")
                    st.session_state["show_login"] = True
                    st.rerun()
                else:
                    st.error(message)
    
    st.stop()

# Main app (only shown if authenticated)
user = get_current_user()

st.markdown("""
<style>
.big-title {
    font-size: 3.5rem; 
    font-weight: bold; 
    color: #1f77b4; 
    text-align: center; 
    margin-bottom: 1rem;
    line-height: 1.2;
}
.subtitle {
    font-size: 1.8rem; 
    color: #666; 
    text-align: center; 
    margin-bottom: 2.5rem;
    font-weight: 500;
}

.problem-card {
    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    min-height: 140px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    transition: transform 0.2s;
}
.problem-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 6px 12px rgba(0,0,0,0.15);
}
.problem-card h4 {
    color: #ffffff;
    font-size: 1.2rem;
    margin-bottom: 0.5rem;
    font-weight: 600;
}
.problem-card p {
    color: #ffffff;
    font-size: 0.95rem;
    line-height: 1.5;
    margin: 0;
}

.solution-card {
    background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    min-height: 140px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    transition: transform 0.2s;
}
.solution-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 6px 12px rgba(0,0,0,0.15);
}
.solution-card h4 {
    color: #ffffff;
    font-size: 1.2rem;
    margin-bottom: 0.5rem;
    font-weight: 600;
}
.solution-card p {
    color: #ffffff;
    font-size: 0.95rem;
    line-height: 1.5;
    margin: 0;
}

.feature-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 12px;
    margin: 1rem 0;
    min-height: 240px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    transition: transform 0.2s;
}
.feature-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 6px 12px rgba(0,0,0,0.15);
}
.feature-card h3 {
    color: #ffffff;
    font-size: 1.5rem;
    margin-bottom: 1rem;
    font-weight: 600;
}
.feature-card p {
    color: #ffffff;
    font-size: 1rem;
    line-height: 1.8;
    margin: 0.3rem 0;
}

.feature-card-green {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
}

.feature-card-blue {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
}

.tech-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    min-height: 200px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    transition: transform 0.2s;
    color: white;
}
.tech-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 6px 12px rgba(0,0,0,0.15);
}
.tech-card h4 {
    color: #ffffff;
    font-size: 1.3rem;
    margin-bottom: 1rem;
    font-weight: 600;
}
.tech-card p {
    color: #ffffff;
    font-size: 1rem;
    line-height: 1.8;
    margin: 0.3rem 0;
}
.tech-card-orange {
    background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
}
.tech-card-teal {
    background: linear-gradient(135deg, #30cfd0 0%, #330867 100%);
}
.tech-card-purple {
    background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
}

.problem-card a, .solution-card a, .feature-card a, .tech-card a {
    text-decoration: none;
    pointer-events: none;
}
.problem-card h4::after, .solution-card h4::after, .feature-card h3::after, .tech-card h4::after {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# User info and logout
col1, col2 = st.columns([6, 1])
with col1:
    st.markdown(f'<p class="big-title">🎯 Unified Customer Intelligence Platform</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="subtitle">Welcome, {user["name"]}!</p>', unsafe_allow_html=True)
with col2:
    if st.button("Logout"):
        logout()
        st.rerun()

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🚨 Business Problems These Solve")
    
    st.markdown("""
    <div class="problem-card">
    <h4>📉 Customer Churn</h4>
    <p>Losing customers costs 5-25x more than retention. Identify at-risk customers before they leave.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="problem-card">
    <h4>💰 Revenue Uncertainty</h4>
    <p>Unable to predict customer lifetime value leads to poor resource allocation and marketing spend.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="problem-card">
    <h4>🎯 Generic Marketing</h4>
    <p>One-size-fits-all campaigns waste budget. Need intelligent customer segmentation for targeted strategies.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("### ✅ Solutions")
    
    st.markdown("""
    <div class="solution-card">
    <h4>🔮 Churn Prediction</h4>
    <p>ML models predict churn probability with SHAP explainability. Take proactive retention actions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="solution-card">
    <h4>📊 CLV Forecasting</h4>
    <p>Predict customer lifetime value to prioritize high-value customers and optimize acquisition costs.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="solution-card">
    <h4>🎨 Smart Segmentation</h4>
    <p>K-Means clustering groups customers by behavior. Create personalized campaigns for each segment.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("### 🚀 Platform Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
    <h3>🧠 Churn Intelligence</h3>
    <p>• Real-time churn prediction</p>
    <p>• SHAP explainability</p>
    <p>• Batch scoring via CSV</p>
    <p>• Stacking ensemble models</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card feature-card-green">
    <h3>💎 CLV Forecasting</h3>
    <p>• Lifetime value prediction</p>
    <p>• Revenue optimization</p>
    <p>• Customer prioritization</p>
    <p>• Batch processing</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card feature-card-blue">
    <h3>🎯 Segmentation</h3>
    <p>• K-Means clustering</p>
    <p>• PCA visualization</p>
    <p>• Cluster profiling</p>
    <p>• Behavioral insights</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("### 🛠️ Technical Stack")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="tech-card">
    <h4>🤖 ML Models</h4>
    <p>• XGBoost</p>
    <p>• Stacking</p>
    <p>• K-Means</p>
    <p>• PCA</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="tech-card tech-card-orange">
    <h4>⚙️ Backend</h4>
    <p>• FastAPI</p>
    <p>• Pydantic</p>
    <p>• Joblib</p>
    <p>• SHAP</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="tech-card tech-card-teal">
    <h4>🎨 Frontend</h4>
    <p>• Streamlit</p>
    <p>• Plotly</p>
    <p>• Pandas</p>
    <p>• NumPy</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="tech-card tech-card-purple">
    <h4>💾 Data</h4>
    <p>• Parquet</p>
    <p>• ETL Pipeline</p>
    <p>• Feature Store</p>
    <p>• Gold Tables</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.success("👈 **Get Started:** Use the sidebar to navigate to Churn Intelligence, CLV Forecasting, or Customer Segmentation modules.")
