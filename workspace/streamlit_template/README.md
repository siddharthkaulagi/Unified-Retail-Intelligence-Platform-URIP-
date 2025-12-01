# Unified Retail Intelligence Platform (URIP)
## AI-Driven Supply Chain Optimization & Forecasting System

### 1. Project Title Variations
*   **Unified Retail Intelligence Platform (URIP): An AI-Driven Approach to Supply Chain Optimization**
*   **Next-Gen Retail Analytics: Integrated Demand Forecasting, GIS Location Intelligence, and Facility Layout Optimization**
*   **Intelligent Retail Ecosystem: Leveraging Machine Learning and Generative AI for Strategic Decision Making**
*   **Smart Supply Chain Management: A Comprehensive Platform for Retail Forecasting and Operational Efficiency**
*   **Data-Driven Retail Transformation: Advanced Analytics for Demand, Customer, and Location Intelligence**

---

### 2. Project Overview
The **Unified Retail Intelligence Platform (URIP)** is a comprehensive, state-of-the-art software solution designed to revolutionize retail operations through the power of Artificial Intelligence, Machine Learning, and Geospatial Analytics. In the highly competitive retail landscape, businesses struggle with fragmented data, inaccurate demand forecasts, inefficient inventory management, and suboptimal store locations. URIP addresses these challenges by providing a centralized, intelligent dashboard that integrates multiple critical modules into a single, cohesive ecosystem.

**Key Problems Solved:**
*   **Demand Uncertainty:** Traditional forecasting methods often fail to capture complex patterns. URIP employs advanced ML models (XGBoost, LSTM, Prophet) to predict sales with high accuracy.
*   **Operational Inefficiency:** Poor facility layouts lead to bottlenecks. URIP uses the Activity Relationship Chart (ARC) methodology to optimize warehouse and store layouts.
*   **Location Blindness:** Choosing store locations without data is risky. URIP's GIS module analyzes demographic and competitor data to recommend optimal expansion sites.
*   **Customer Churn:** Identifying at-risk customers is difficult. URIP's CRM module segments customers and predicts churn using RFM analysis and clustering.

**Key Features:**
*   **Multi-Model Forecasting Engine:** Auto-selection of the best model (ARIMA, Prophet, LSTM, etc.).
*   **Geospatial Intelligence:** Interactive maps for store location strategy using KML ward data.
*   **Facility Layout Optimization:** Automated generation of optimal floor plans.
*   **Generative AI Integration:** Google Gemini-powered chatbot and report generation for actionable insights.
*   **Comprehensive CRM:** Customer segmentation, CLV prediction, and churn risk analysis.

---

### 3. Full Tech Stack

#### **Core Framework & Language**
*   **Python 3.9+**: The primary programming language, chosen for its rich ecosystem of data science libraries.
*   **Streamlit**: Used for the Presentation Layer to create a responsive, interactive web-based user interface.

#### **Data Processing & Analysis**
*   **Pandas & NumPy**: For high-performance data manipulation, cleaning, and numerical computations.
*   **OpenPyXL**: For reading and writing Excel files, essential for report generation.

#### **Machine Learning & Forecasting**
*   **Scikit-Learn**: Used for Random Forest, K-Means Clustering, and data preprocessing (StandardScaler).
*   **XGBoost / LightGBM**: Gradient boosting frameworks for high-performance regression and classification tasks.
*   **Prophet**: Facebook's time-series forecasting tool, robust to missing data and shifts in trend.
*   **Statsmodels (ARIMA/SARIMA)**: For classical statistical time-series analysis.
*   **TensorFlow / Keras (LSTM)**: Deep learning libraries for Long Short-Term Memory networks to capture sequential dependencies.

#### **Geospatial Analysis (GIS)**
*   **Folium**: For rendering interactive leaflet maps.
*   **GeoPandas**: For handling geospatial data (Shapefiles, GeoJSON).
*   **Shapely**: For geometric operations (points, polygons).
*   **Fiona**: For reading and writing spatial data formats like KML.
*   **Geopy**: For geodesic distance calculations between coordinates.

#### **Optimization & Operations Research**
*   **NetworkX**: For graph theory and network analysis in facility layout (relationship diagrams).
*   **SciPy / PuLP**: For mathematical optimization tasks.

#### **Generative AI & NLP**
*   **Google Generative AI (Gemini API)**: Powers the intelligent chatbot and automated report analysis.
*   **Python-Docx**: For generating Word documents with AI insights.

#### **Visualization**
*   **Plotly Express & Graph Objects**: For interactive, publication-quality charts (Sankey, Scatter, Line, Bar).
*   **Matplotlib / Seaborn**: For static statistical visualizations.

#### **Database & State Management**
*   **SQLite**: Lightweight relational database for user authentication and session management.
*   **Streamlit Session State**: For managing application state across re-runs.

---

### 4. System Architecture

The system follows a modular **Layered Architecture**, ensuring separation of concerns and scalability.

```ascii
+---------------------------------------------------------------+
|                   Presentation Layer (Streamlit UI)           |
|  [Dashboard] [Upload] [Forecasting] [CRM] [GIS] [Layout] [AI] |
+-------------------------------+-------------------------------+
                                |
                                v
+---------------------------------------------------------------+
|                      Application Layer                        |
|  [Auth Manager] [Session Manager] [Page Routing] [State Mgmt] |
+-------------------------------+-------------------------------+
                                |
                                v
+---------------------------------------------------------------+
|                       Logic / ML Layer                        |
|  [Forecasting Engine]    [GIS Processor]    [Layout Optimizer]|
|  (Prophet, XGB, LSTM)    (GeoPandas, Folium) (NetworkX, ARC)  |
|                                                               |
|  [CRM Analytics]         [AI Assistant]                       |
|  (K-Means, RFM)          (Gemini API)                         |
+-------------------------------+-------------------------------+
                                |
                                v
+---------------------------------------------------------------+
|                         Data Layer                            |
|  [SQLite DB] [CSV/Excel Files] [KML/GeoJSON] [Pickle Models]  |
+---------------------------------------------------------------+
```

1.  **Presentation Layer**: The user interacts with the Streamlit frontend. It handles file uploads, parameter inputs, and displays visualizations.
2.  **Application Layer**: Manages user sessions, authentication checks, and routing between different modules (pages).
3.  **Logic/ML Layer**: The core "brain" of the system. It contains the algorithms for forecasting, optimization, and spatial analysis.
4.  **Data Layer**: Handles persistent storage (User DB) and temporary data storage (Session State, uploaded files).

---

### 5. Directory / File Structure

```text
Retail Sales Prediction/
├── app.py                      # Main entry point (Login & Navigation)
├── requirements.txt            # Python dependencies
├── assets/                     # Static assets (CSS, Images)
│   ├── custom.css              # Custom styling for Streamlit
│   └── urip_logo.png           # Project Logo
├── pages/                      # Application Modules
│   ├── 01_Upload_Data.py       # Data ingestion module
│   ├── 02_Model_Selection.py   # Model configuration
│   ├── 03_Dashboard.py         # Main analytics dashboard
│   ├── 04_Reports.py           # Report generation & download
│   ├── 05_Settings.py          # User settings
│   ├── 06_Demand_Forecasting.py# Detailed forecast visualization
│   ├── 07_CRM_Analytics.py     # Customer segmentation & churn
│   ├── 08_Facility_Layout.py   # Warehouse layout optimization
│   ├── 09_AI_Chatbot.py        # Gemini-powered assistant
│   └── 10_Store_Location_GIS.py# GIS & Location Intelligence
├── utils/                      # Helper Functions & Logic
│   ├── database.py             # SQLite connection & User Mgmt
│   ├── feature_flags.py        # Feature toggles
│   ├── floating_chatbot.py     # UI component for chatbot
│   ├── gemini_ai.py            # Gemini API integration wrapper
│   ├── models.py               # ML Forecasting Model classes
│   └── ui_components.py        # Reusable UI widgets (Sidebar)
├── data/                       # (Optional) Data storage
│   ├── bbmp_final_new_wards.kml# Bangalore Ward Boundaries
│   └── sample_retail_data.csv  # Demo dataset
└── users.db                    # SQLite User Database
```

---

### 6. Detailed Module-by-Module Explanation

#### **1. Data Upload & Preprocessing (`01_Upload_Data.py`)**
*   **Purpose**: Ingests raw retail data (CSV/Excel) for analysis.
*   **Key Functions**: Validates schema, handles missing values, and converts date columns.
*   **Output**: Cleaned DataFrame stored in `st.session_state`.

#### **2. Model Selection (`02_Model_Selection.py`)**
*   **Purpose**: Allows users to configure forecasting parameters.
*   **Features**: Select target columns, date columns, and specific algorithms (Prophet, ARIMA, etc.) to run.

#### **3. Demand Forecasting (`06_Demand_Forecasting.py` & `utils/models.py`)**
*   **Purpose**: The core forecasting engine.
*   **Logic**: Trains selected models on historical data.
*   **Visualizations**: Interactive time-series plots showing Historical vs. Forecasted data with confidence intervals.

#### **4. CRM Analytics (`07_CRM_Analytics.py`)**
*   **Purpose**: Analyzes customer behavior.
*   **Logic**:
    *   **RFM Analysis**: Segments customers into "Champions", "At Risk", etc.
    *   **CLV Prediction**: Estimates future value of customers.
    *   **Churn Prediction**: Identifies customers likely to stop purchasing.
*   **Visualizations**: Scatter plots (Recency vs Monetary), Pie charts of segments.

#### **5. Store Location GIS (`10_Store_Location_GIS.py`)**
*   **Purpose**: Strategic site selection for new stores.
*   **Data**: Uses KML files for Bangalore Ward boundaries and population data.
*   **Logic**: Scores locations based on Population Density, Competitor Proximity, and Accessibility.
*   **Visualizations**: Folium maps with heatmaps, marker clusters, and choropleth layers.

#### **6. Facility Layout Optimization (`08_Facility_Layout.py`)**
*   **Purpose**: Optimizes physical space in warehouses/stores.
*   **Logic**: Uses **Activity Relationship Chart (ARC)** input to generate a grid-based layout that minimizes distance between related departments.
*   **Visualizations**: Network graphs (NetworkX) and 2D Layout diagrams (Plotly).

#### **7. AI Chatbot (`09_AI_Chatbot.py` & `utils/gemini_ai.py`)**
*   **Purpose**: Natural language interface for the platform.
*   **Logic**: Sends user queries + context (forecast data) to Google Gemini API.
*   **Features**: Can analyze uploaded images (charts) and generate text summaries.

---

### 7. Machine Learning Models

#### **1. Prophet**
*   **Type**: Additive Regression Model.
*   **Use Case**: Best for data with strong seasonal effects and missing data.
*   **Formula**: `y(t) = g(t) + s(t) + h(t) + ε_t`
    *   `g(t)`: Trend function (linear/logistic).
    *   `s(t)`: Seasonality (Fourier series).
    *   `h(t)`: Holiday effects.

#### **2. ARIMA (AutoRegressive Integrated Moving Average)**
*   **Type**: Statistical Linear Model.
*   **Use Case**: Short-term forecasting for stationary data.
*   **Components**: `AR(p)` (past values), `I(d)` (differencing), `MA(q)` (past errors).

#### **3. XGBoost / LightGBM**
*   **Type**: Gradient Boosted Decision Trees.
*   **Use Case**: High accuracy, handles non-linear relationships well.
*   **Feature Engineering**: Creates lag features (sales_lag_7), rolling means, and date parts (day_of_week, month).

#### **4. LSTM (Long Short-Term Memory)**
*   **Type**: Recurrent Neural Network (RNN).
*   **Use Case**: Captures long-term dependencies in sequential data.
*   **Architecture**: Input Layer -> LSTM Layer (50 units) -> Dropout -> Dense Output.

#### **5. K-Means Clustering (CRM)**
*   **Use Case**: Customer Segmentation.
*   **Logic**: Partitions customers into `k` clusters by minimizing within-cluster variance (Euclidean distance).

---

### 8. GIS / Mapping Logic

*   **KML Parsing**: Uses `fiona` and `xml.etree` to parse complex KML files containing Ward boundaries and extended data (Population).
*   **Ward Scoring Algorithm**:
    *   `Score = (Pop_Score * 0.4) + (Dist_Score * 0.3) + (Access_Score * 0.2) + (Socio_Score * 0.1)`
    *   Calculates geodesic distance to nearest competitor using `geopy`.
*   **Heatmaps**: Generated using `folium.plugins.HeatMap` based on normalized population density.

---

### 9. Facility Layout Optimization

*   **ARC Matrix**: User defines relationships (A, E, I, O, U, X) between departments.
    *   **A**: Absolutely Necessary (Score: 10)
    *   **X**: Undesirable (Score: -10)
*   **Optimization Logic**:
    1.  Calculates a "Closeness Score" for every pair.
    2.  Generates a grid layout.
    3.  Evaluates layout efficiency: `Σ (Relationship_Weight / Distance)`.
    4.  (Future: Genetic Algorithm can be applied here).

---

### 10. AI Chatbot

*   **Prompt Engineering**: Context-aware prompts are constructed.
    *   *System Prompt*: "You are an AI retail forecasting assistant..."
    *   *Context Injection*: Current forecast data is appended to the prompt.
*   **API Flow**: User Input -> `GeminiChatbot.chat()` -> Google Gemini API -> Response -> UI Display.

---

### 11. Database / Data Storage

*   **SQLite (`users.db`)**:
    *   Table `users`: Stores `id`, `username`, `password_hash`, `email`, `role`.
    *   Table `sessions`: Manages active login tokens.
*   **Session State**: Streamlit's `st.session_state` is used as a volatile in-memory store for:
    *   `uploaded_data`: The raw dataframe.
    *   `model_results`: Dictionary of trained models and forecasts.
    *   `gis_data`: Loaded GeoDataFrames.

---

### 12. Application Flow / User Journey

1.  **Login**: User authenticates via the Login Page.
2.  **Data Upload**: User uploads a sales CSV file. System validates and previews it.
3.  **Model Config**: User selects "Sales" as target and chooses "Prophet" and "XGBoost".
4.  **Forecasting**: System trains models, generates predictions, and displays the "Forecast vs Actual" chart.
5.  **Analysis**:
    *   User checks **CRM** tab to see "At Risk" customers.
    *   User checks **GIS** tab to find a location for a new store to capture lost demand.
6.  **Reporting**: User asks the **Chatbot** "Summarize the sales trend" and downloads a PDF report.

---

### 13. Installation Guide

**Prerequisites:**
*   Python 3.9 or higher
*   Git

**Step 1: Clone the Repository**
```bash
git clone https://github.com/your-repo/retail-intelligence-platform.git
cd retail-intelligence-platform
```

**Step 2: Create Virtual Environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

**Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Step 4: Set API Keys**
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

**Step 5: Run the Application**
```bash
streamlit run app.py
```

---

### 14. Future Enhancements

*   **Real-Time POS Integration**: Direct API connection to Point-of-Sale systems for live data.
*   **Reinforcement Learning**: For dynamic pricing and inventory replenishment policies.
*   **Mobile Application**: A Flutter/React Native companion app for store managers.
*   **Blockchain**: For supply chain transparency and provenance tracking.

---

### 15. License

This project is licensed under the **MIT License**.

---
*Generated for Academic Project Submission*
