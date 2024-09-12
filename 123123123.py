# Streamlit app template for common tasks
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Title of the app
st.title("Software Technology - Streamlit App")

# Sidebar for selecting tasks
task = st.sidebar.selectbox("Select Task",
                            ["File Upload",
                             "Data Analysis",
                             "Data Visualization",
                             "Feature Selection",
                             "Classification Algorithms",
                             "Information"])

# Task 1: File Upload
if task == "File Upload":
    st.header("File Upload")
    uploaded_file = st.file_uploader("Upload a CSV File", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:", df.head())

# Task 2: Data Analysis
elif task == "Data Analysis":
    st.header("Data Analysis")

    # Upload data for analysis
    uploaded_file = st.file_uploader("Upload a CSV File for Analysis", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Ensure last column is the label and rest are features
        st.write("**Data Specifications:**")
        st.write(f"- Rows (S): {df.shape[0]}, Columns (F+1): {df.shape[1]}")
        st.write("The last column is assumed to be the label (target).")

        # Display data details
        st.write("Basic Info:")
        st.write(df.describe())  # Shows basic statistics of the data

        # Information about data structure
        st.write("Data Structure:")
        st.write(df.info())  # Provides information about data types, nulls

# Task 3: Data Visualization
elif task == "Data Visualization":
    st.header("Data Visualization")
    uploaded_file = st.file_uploader("Upload a CSV File for Plotting", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Columns in dataset:", df.columns.tolist())
        x_col = st.selectbox("Choose X-axis column", df.columns)
        y_col = st.selectbox("Choose Y-axis column", df.columns)
        plt.figure(figsize=(10, 6))
        plt.scatter(df[x_col], df[y_col])
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"Scatter plot of {x_col} vs {y_col}")
        st.pyplot(plt)

# Task 4: Feature Selection (Machine Learning Tab 1)
elif task == "Feature Selection":
    st.header("Feature Selection")

    # Upload data
    uploaded_file = st.file_uploader("Upload a CSV File for Feature Selection", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        X = df.iloc[:, :-1]  # Features (all columns except last)
        y = df.iloc[:, -1]  # Labels (last column)

        # Select number of features
        num_features = st.slider("Select number of features to keep", 1, X.shape[1], 5)

        # Feature selection using ANOVA F-test
        selector = SelectKBest(f_classif, k=num_features)
        X_new = selector.fit_transform(X, y)

        # Create a dataframe with selected features
        selected_features = pd.DataFrame(X_new, columns=X.columns[selector.get_support()])
        st.write("Dataset with reduced features:")
        st.write(selected_features.head())

# Task 5: Classification Algorithms (Machine Learning Tab 2)
elif task == "Classification Algorithms":
    st.header("Classification Algorithms")

    # Upload data
    uploaded_file = st.file_uploader("Upload a CSV File for Classification", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        X = df.iloc[:, :-1]  # Features (all columns except last)
        y = df.iloc[:, -1]   # Labels (last column)

        # Encode labels if needed
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Optionally perform feature selection
        apply_feature_selection = st.checkbox("Apply Feature Selection")
        if apply_feature_selection:
            num_features = st.slider("Select number of features to keep", 1, X_train.shape[1], 5)
            selector = SelectKBest(f_classif, k=num_features)
            X_train = selector.fit_transform(X_train, y_train)
            X_test = selector.transform(X_test)  # Apply the same transformation to the test set

        # Standardize the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Classifier 1: KNN
        st.subheader("K-Nearest Neighbors (KNN)")
        k = st.slider("Select value of k for KNN", 1, 20, 5)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)

        # Metrics for KNN
        acc_knn = accuracy_score(y_test, y_pred_knn)
        f1_knn = f1_score(y_test, y_pred_knn, average='weighted')
        roc_auc_knn = roc_auc_score(y_test, knn.predict_proba(X_test), multi_class='ovr')

        st.write(f"**KNN Accuracy**: {acc_knn:.2f}")
        st.write(f"**KNN F1-Score**: {f1_knn:.2f}")
        st.write(f"**KNN ROC-AUC**: {roc_auc_knn:.2f}")

        # Classifier 2: Random Forest
        st.subheader("Random Forest")
        n_estimators = st.slider("Select number of estimators for Random Forest", 10, 200, 100)
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)

        # Metrics for Random Forest
        acc_rf = accuracy_score(y_test, y_pred_rf)
        f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
        roc_auc_rf = roc_auc_score(y_test, rf.predict_proba(X_test), multi_class='ovr')

        st.write(f"**Random Forest Accuracy**: {acc_rf:.2f}")
        st.write(f"**Random Forest F1-Score**: {f1_rf:.2f}")
        st.write(f"**Random Forest ROC-AUC**: {roc_auc_rf:.2f}")

        # Results comparison
        st.subheader("Comparison of KNN and Random Forest")
        comparison_df = pd.DataFrame({
            'Metric': ['Accuracy', 'F1-Score', 'ROC-AUC'],
            'KNN': [acc_knn, f1_knn, roc_auc_knn],
            'Random Forest': [acc_rf, f1_rf, roc_auc_rf]
        })
        st.write(comparison_df)


# Info Tab
elif task == "Information":
    st.header("Information")
    st.write("Team Members: George Anagnostou")
    st.write("George Anagnostou: Sole creator")

