import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import umap_ as UMAP
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer

# Title of the application
st.title("Data Mining and Analysis Application")

# Data Loading
st.header("Data Loading")
uploaded_files = st.file_uploader("Choose two files", type=["csv", "xlsx", "tsv"], accept_multiple_files=True)
if len(uploaded_files) == 2:
    data1 = None
    data2 = None

    if uploaded_files[0].name.endswith('csv'):
        data1 = pd.read_csv(uploaded_files[0])
    elif uploaded_files[0].name.endswith('xlsx'):
        data1 = pd.read_excel(uploaded_files[0])
    else:
        data1 = pd.read_csv(uploaded_files[0], sep='\t')

    if uploaded_files[1].name.endswith('csv'):
        data2 = pd.read_csv(uploaded_files[1])
    elif uploaded_files[1].name.endswith('xlsx'):
        data2 = pd.read_excel(uploaded_files[1])
    else:
        data2 = pd.read_csv(uploaded_files[1], sep='\t')

    st.write("Data Loaded Successfully!")
    st.write("Dataset 1:")
    st.dataframe(data1)
    st.write("Dataset 2:")
    st.dataframe(data2)

    # Explicitly cast numeric columns
    def ensure_numeric(df):
        numeric_df = df.apply(pd.to_numeric, errors='coerce')
        return numeric_df

    data1_numeric = ensure_numeric(data1)
    data2_numeric = ensure_numeric(data2)

    # Dataset selection
    dataset_choice = st.selectbox("Choose dataset to analyze", ["Dataset 1", "Dataset 2"])
    if dataset_choice == "Dataset 1":
        data = data1_numeric
    else:
        data = data2_numeric

    st.write(f"Selected {dataset_choice} for analysis.")
    st.dataframe(data)

    # Preprocessing: Select only numeric columns
    numeric_data = data.select_dtypes(include=[np.number])
    num_features = numeric_data.shape[1]

    st.write("Numeric Data for Analysis:")
    st.dataframe(numeric_data)
    st.write(f"Number of Numeric Features: {num_features}")

    # Define X and y
    X = numeric_data
    y = data.iloc[:, -1]  # Assuming the last column is the label

    # Handle missing values by imputation
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Visualization Tab
    st.header("Visualization Tab")

    # PCA Visualization
    st.subheader("PCA Visualization")
    pca_option = st.selectbox("PCA Dimension", ["2D", "3D"])
    if st.button("Generate PCA Visualization"):
        if num_features >= 2:
            if pca_option == "2D":
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(X_imputed)
                plt.figure(figsize=(10, 6))
                plt.scatter(pca_result[:, 0], pca_result[:, 1], c=y, cmap='viridis')
                plt.title("PCA Result - 2D")
                plt.xlabel("Principal Component 1")
                plt.ylabel("Principal Component 2")
                st.pyplot(plt)
            elif pca_option == "3D":
                if num_features >= 3:
                    pca = PCA(n_components=3)
                    pca_result = pca.fit_transform(X_imputed)
                    fig = plt.figure(figsize=(10, 7))
                    ax = fig.add_subplot(111, projection='3d')
                    scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], c=y, cmap='viridis')
                    plt.title("PCA Result - 3D")
                    ax.set_xlabel("Principal Component 1")
                    ax.set_ylabel("Principal Component 2")
                    ax.set_zlabel("Principal Component 3")
                    fig.colorbar(scatter)
                    st.pyplot(fig)
                else:
                    st.write("Not enough features for 3D PCA. Ensure your data has at least 3 numeric features.")
        else:
            st.write("Not enough features for PCA. Ensure your data has at least 2 numeric features.")

    # UMAP Visualization
    st.subheader("UMAP Visualization")
    umap_option = st.selectbox("UMAP Dimension", ["2D", "3D"])
    if st.button("Generate UMAP Visualization"):
        if num_features >= 2:
            if umap_option == "2D":
                umap_model = UMAP(n_components=2)
                umap_result = umap_model.fit_transform(X_imputed)
                plt.figure(figsize=(10, 6))
                plt.scatter(umap_result[:, 0], umap_result[:, 1], c=y, cmap='viridis')
                plt.title("UMAP Result - 2D")
                plt.xlabel("UMAP Component 1")
                plt.ylabel("UMAP Component 2")
                st.pyplot(plt)
            elif umap_option == "3D":
                if num_features >= 3:
                    umap_model = UMAP(n_components=3)
                    umap_result = umap_model.fit_transform(X_imputed)
                    fig = plt.figure(figsize=(10, 7))
                    ax = fig.add_subplot(111, projection='3d')
                    scatter = ax.scatter(umap_result[:, 0], umap_result[:, 1], umap_result[:, 2], c=y, cmap='viridis')
                    plt.title("UMAP Result - 3D")
                    ax.set_xlabel("UMAP Component 1")
                    ax.set_ylabel("UMAP Component 2")
                    ax.set_zlabel("UMAP Component 3")
                    fig.colorbar(scatter)
                    st.pyplot(fig)
                else:
                    st.write("Not enough features for 3D UMAP. Ensure your data has at least 3 numeric features.")
        else:
            st.write("Not enough features for UMAP. Ensure your data has at least 2 numeric features.")

    # EDA Charts
    st.header("Exploratory Data Analysis (EDA) Charts")

    # Histogram of numeric features
    st.subheader("Histogram of Numeric Features")
    num_features_to_plot = st.multiselect("Select Numeric Features for Histogram", numeric_data.columns)
    if num_features_to_plot:
        for feature in num_features_to_plot:
            plt.figure(figsize=(10, 6))
            sns.histplot(numeric_data[feature], kde=True)
            plt.title(f"Histogram of {feature}")
            plt.xlabel(feature)
            plt.ylabel("Frequency")
            st.pyplot(plt)

    # Pairplot of numeric features
    st.subheader("Pairplot of Numeric Features")
    if st.button("Generate Pairplot"):
        if num_features >= 2:
            pairplot_data = pd.concat([numeric_data, y.rename('target')], axis=1)
            pairplot = sns.pairplot(pairplot_data, hue='target')
            st.pyplot(pairplot)
        else:
            st.write("Not enough features for Pairplot. Ensure your data has at least 2 numeric features.")

    # Machine Learning Tabs
    st.header("Machine Learning Tabs")
    feature_selection_tab = st.selectbox("Select Feature Selection Method", ["SelectKBest"])
    num_features_to_select = st.number_input("Number of Features to Select", min_value=1, max_value=num_features)

    if st.button("Run Feature Selection"):
        if feature_selection_tab == "SelectKBest":
            selector = SelectKBest(score_func=f_classif, k=num_features_to_select)
            X_selected = selector.fit_transform(X_imputed, y)
            st.write("Selected Features:")
            st.dataframe(pd.DataFrame(X_selected, columns=[f'Feature {i+1}' for i in range(X_selected.shape[1])]))

    # Classification Algorithms
    st.header("Classification Algorithms")
    algorithm = st.selectbox("Select Algorithm", ["KNN"])
    k_value = st.number_input("Enter the value of k for KNN", min_value=1)

    if st.button("Run Classification"):
        if algorithm == "KNN":
            X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
            knn = KNeighborsClassifier(n_neighbors=k_value)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            if len(np.unique(y_test)) == 2:  # Check if binary classification
                roc_auc = roc_auc_score(y_test, knn.predict_proba(X_test)[:, 1])
                st.write(f"ROC AUC: {roc_auc}")
            else:
                st.write("ROC AUC is not applicable for multiclass classification.")

            st.write(f"Accuracy: {accuracy}")
            st.write(f"F1 Score: {f1}")

# Info Tab
st.header("Information")
st.write("This application was developed by the Data Science Team.")
st.write("Team Members:")
st.write("- Member 1: Data Loading and Visualization")
st.write("- Member 2: Machine Learning Implementation")
st.write("- Member 3: Application Design and Deployment")
