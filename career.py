import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats
import plotly.express as px
# Add background image with transparent overlay
page_bg_img = '''

'''
st.markdown(page_bg_img, unsafe_allow_html=True)


st.set_page_config(page_title="StatAnveshak", layout="wide")
st.title("📊 StatAnveshak - Statistical Analysis & Learning Tool")

# Sidebar Layout
st.sidebar.header("📁 File Upload")
uploaded_file = st.sidebar.file_uploader("Upload CSV/Excel file", type=['csv', 'xlsx'])

if uploaded_file:
    # File Reading
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.sidebar.success("File uploaded successfully!")
    st.sidebar.write("Shape:", df.shape)

    # Tabs for features
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📄 Dataset",
        "📊 Descriptive Stats",
        "📈 Visualizations",
        "🔍 Inferential Stats",
        "🧠 ML Models",
        "📘 Learn"
    ])

    with tab1:
        st.subheader("Dataset Viewer")
        st.dataframe(df, use_container_width=True)
        if st.checkbox("Show column types"):
            st.write(df.dtypes)

    with tab2:
        st.subheader("Descriptive Statistics")
        st.write(df.describe())
        if st.checkbox("Show missing values count"):
            st.write(df.isnull().sum())

    with tab3:
        st.subheader("Visualizations")
        chart_type = st.selectbox("Choose chart type", [
            "Histogram", "Box Plot", "Scatter Plot", "Heatmap", "Pair Plot", "Line Chart"
        ])

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

        if chart_type == "Histogram":
            col = st.selectbox("Select Column", numeric_cols)
            bins = st.slider("Number of Bins", 5, 100, 20)
            fig = px.histogram(df, x=col, nbins=bins)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Box Plot":
            col = st.selectbox("Numeric Column", numeric_cols)
            by = st.selectbox("Group by (optional)", [None] + cat_cols)
            fig = px.box(df, x=by, y=col) if by else px.box(df, y=col)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Scatter Plot":
            x = st.selectbox("X-axis", numeric_cols)
            y = st.selectbox("Y-axis", numeric_cols, index=1)
            fig = px.scatter(df, x=x, y=y, trendline="ols")
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Heatmap":
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=True)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Pair Plot":
            st.info("Showing first 200 rows for speed.")
            fig = sns.pairplot(df[numeric_cols].dropna().sample(min(200, len(df))))
            st.pyplot(fig)

        elif chart_type == "Line Chart":
            col = st.selectbox("Select Time Series Column", numeric_cols)
            st.line_chart(df[col])

    with tab4:
        st.subheader("Inferential Statistics")
        test_type = st.selectbox("Choose Test", ["T-Test", "ANOVA"])
        target = st.selectbox("Select Target Column", df.select_dtypes(include=np.number).columns)
        group_col = st.selectbox("Group by Column", df.select_dtypes(exclude=np.number).columns)

        if test_type == "T-Test":
            groups = df[group_col].dropna().unique()
            if len(groups) == 2:
                vals1 = df[df[group_col] == groups[0]][target].dropna()
                vals2 = df[df[group_col] == groups[1]][target].dropna()
                t_stat, p_val = stats.ttest_ind(vals1, vals2)
                st.write(f"**T-Statistic:** {t_stat:.3f}, **P-value:** {p_val:.4f}")
            else:
                st.warning("T-Test requires exactly 2 groups.")

        elif test_type == "ANOVA":
            grouped = [df[df[group_col] == val][target].dropna() for val in df[group_col].unique()]
            f_stat, p_val = stats.f_oneway(*grouped)
            st.write(f"**F-Statistic:** {f_stat:.3f}, **P-value:** {p_val:.4f}")

    with tab5:
        st.subheader("ML & Dimensionality Reduction")
        ml_option = st.radio("Choose Model", ["K-Means Clustering", "PCA", "Linear Regression"])
        features = st.multiselect("Select Features", df.select_dtypes(include=np.number).columns.tolist())

        if len(features) > 1:
            X = df[features].dropna()
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            if ml_option == "K-Means Clustering":
                k = st.slider("Number of Clusters (K)", 2, 10, 3)
                km = KMeans(n_clusters=k, n_init=10)
                y_km = km.fit_predict(X_scaled)
                df['Cluster'] = y_km
                fig = px.scatter(df, x=features[0], y=features[1], color=y_km.astype(str))
                st.plotly_chart(fig)

            elif ml_option == "PCA":
                pca = PCA(n_components=2)
                pca_comp = pca.fit_transform(X_scaled)
                df['PC1'], df['PC2'] = pca_comp[:, 0], pca_comp[:, 1]
                fig = px.scatter(df, x="PC1", y="PC2")
                st.plotly_chart(fig)

            elif ml_option == "Linear Regression":
                target = st.selectbox("Select Target (Y)", features)
                X_lr = X.drop(columns=target)
                y_lr = X[target]
                X_train, X_test, y_train, y_test = train_test_split(X_lr, y_lr, test_size=0.3, random_state=42)
                model = LinearRegression().fit(X_train, y_train)
                score = model.score(X_test, y_test)
                st.write(f"**R² Score:** {score:.3f}")
                st.write("**Coefficients:**")
                st.write(dict(zip(X_train.columns, model.coef_)))

    with tab6:
        st.subheader("📘 Learning & Tutorials")
        st.info("This section will include guided lessons, quizzes, and real-world examples in future updates.")
        st.markdown("""
        - ✅ Understand visualization types  
        - ✅ Interactive examples  
        - ⏳ Quizzes (Coming Soon)  
        - ⏳ Guided ML Notebooks (Coming Soon)  
        - ⏳ Applied case studies (Coming Soon)  
        """)

else:
    st.info("📂 Upload a dataset (CSV or Excel) from the sidebar to begin.")
