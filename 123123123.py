import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import umap
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

st.title('Εφαρμογή Εξόρυξης και Ανάλυσης Δεδομένων')

# Sidebar for data upload
st.sidebar.header('Φόρτωση Δεδομένων')
uploaded_file = st.sidebar.file_uploader('Επιλέξτε ένα αρχείο CSV/Excel/TSV', type=['csv', 'xlsx', 'tsv'])

if uploaded_file:
    # Load data
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.tsv'):
        df = pd.read_csv(uploaded_file, delimiter='\t')

    st.write('### Δεδομένα')
    st.write(df.head())

    # Check if the dataframe meets the requirements
    if df.shape[1] < 2:
        st.error('Ο πίνακας πρέπει να έχει τουλάχιστον δύο στήλες.')
    else:
        st.sidebar.header('Visualization Tab')
        if st.sidebar.checkbox('Εμφάνιση 2D/3D Visualization'):
            st.write('## Visualization Tab')

            # PCA 2D
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(df.iloc[:, :-1])
            pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
            pca_df['label'] = df.iloc[:, -1]
            fig1 = px.scatter(pca_df, x='PCA1', y='PCA2', color='label')
            st.plotly_chart(fig1, use_container_width=True)

            # UMAP 3D
            reducer = umap.UMAP(n_components=3)
            umap_result = reducer.fit_transform(df.iloc[:, :-1])
            umap_df = pd.DataFrame(data=umap_result, columns=['UMAP1', 'UMAP2', 'UMAP3'])
            umap_df['label'] = df.iloc[:, -1]
            fig2 = px.scatter_3d(umap_df, x='UMAP1', y='UMAP2', z='UMAP3', color='label')
            st.plotly_chart(fig2, use_container_width=True)

        st.sidebar.header('Feature Selection Tab')
        if st.sidebar.checkbox('Εμφάνιση Feature Selection'):
            st.write('## Feature Selection Tab')
            k = st.slider('Επιλέξτε αριθμό χαρακτηριστικών', 1, df.shape[1] - 1, 5)
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            selector = SelectKBest(f_classif, k=k)
            X_new = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()]
            st.write(f'### Επιλεγμένα Χαρακτηριστικά: {selected_features.tolist()}')
            reduced_df = pd.DataFrame(X_new, columns=selected_features)
            reduced_df['label'] = y.values
            st.write(reduced_df.head())

        st.sidebar.header('Classification Tab')
        if st.sidebar.checkbox('Εμφάνιση Classification'):
            st.write('## Classification Tab')
            classifier_name = st.selectbox('Επιλέξτε αλγόριθμο', ['KNN', 'Random Forest'])
            param = st.slider('Επιλέξτε παράμετρο αλγορίθμου', 1, 10, 5)

            # Classification before feature selection
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            if classifier_name == 'KNN':
                classifier = KNeighborsClassifier(n_neighbors=param)
            elif classifier_name == 'Random Forest':
                classifier = RandomForestClassifier(max_depth=param, random_state=42)

            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            roc_auc = roc_auc_score(y_test, classifier.predict_proba(X_test), multi_class='ovr')

            st.write('### Πριν την Επιλογή Χαρακτηριστικών')
            st.write(f'Ακρίβεια: {acc}')
            st.write(f'F1-Score: {f1}')
            st.write(f'ROC-AUC: {roc_auc}')

            # Classification after feature selection
            X_new = reduced_df.iloc[:, :-1]
            y_new = reduced_df.iloc[:, -1]
            X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.3,
                                                                                random_state=42)

            classifier.fit(X_train_new, y_train_new)
            y_pred_new = classifier.predict(X_test_new)
            acc_new = accuracy_score(y_test_new, y_pred_new)
            f1_new = f1_score(y_test_new, y_pred_new, average='weighted')
            roc_auc_new = roc_auc_score(y_test_new, classifier.predict_proba(X_test_new), multi_class='ovr')

            st.write('### Μετά την Επιλογή Χαρακτηριστικών')
            st.write(f'Ακρίβεια: {acc_new}')
            st.write(f'F1-Score: {f1_new}')
            st.write(f'ROC-AUC: {roc_auc_new}')

        st.sidebar.header('Info Tab')
        if st.sidebar.checkbox('Εμφάνιση Info Tab'):
            st.write('## Info Tab')
            st.write('Αυτή είναι μια εφαρμογή για εξόρυξη και ανάλυση δεδομένων.')
            st.write('### Ομάδα Ανάπτυξης')
            st.write('Μέλος 1: Υλοποίηση της φόρτωσης δεδομένων και της οπτικοποίησης')
            st.write('Μέλος 2: Υλοποίηση της επιλογής χαρακτηριστικών και της κατηγοριοποίησης')
            st.write('Μέλος 3: Υλοποίηση του Docker και GitHub, σύνταξη της έκθεσης')

# Save the report to LaTeX, UML diagram, and describe the software lifecycle model
st.sidebar.header('Άλλα')
if st.sidebar.button('Δημιουργία Έκθεσης'):
    st.write('Η έκθεση θα δημιουργηθεί και θα αποθηκευτεί.')

if st.sidebar.button('Δημιουργία UML Διαγράμματος'):
    st.write('Το UML διάγραμμα θα δημιουργηθεί και θα αποθηκευτεί.')

if st.sidebar.button('Περιγραφή Κύκλου Ζωής Λογισμικού'):
    st.write('Θα περιγραφεί ο κύκλος ζωής της έκδοσης του λογισμικού.')
