import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, accuracy_score
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Airbnb Brasil", layout="wide")
import streamlit as st
import pandas as pd

@st.cache_resource
def load_data():
    df = pd.read_csv('Limpio_Brasil.csv')

    # Eliminar columna 'Unnamed: 0' si existe
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

    numeric_df = df.select_dtypes(['float', 'int'])
    numeric_cols = numeric_df.columns

    text_df = df.select_dtypes(['object'])
    text_cols = text_df.columns

    return df, numeric_cols, text_cols, numeric_df

# Cargar
df, numeric_cols, text_cols, numeric_df = load_data()


st.markdown("""
<style>
/* Fondo de toda la app */
[data-testid="stAppViewContainer"] {
    background-color: #E8F5E9; /* Verde pastel clarito */
}

/* Fondo del sidebar */
section[data-testid="stSidebar"] {
    background-color: #FFFDE7; /* Amarillo pastel */
}

/* Encabezados */
h1, h2, h3 {
    color: #1B5E20; /* Verde bandera Brasil */
    font-weight: bold;
}

/* Botones personalizados para que se vean mejor sobre cualquier fondo */
div.stButton>button {
    background-color: #1976D2; /* Azul fuerte */
    color: white;
    border-radius: 8px;
    padding: 0.5em 1em;
    font-size: 1em;
    border: none;
    transition: 0.3s ease;
}

div.stButton>button:hover {
    background-color: #1565C0; /* Un azul más oscuro al pasar el mouse */
}

/* Selectbox más bonito */
div.stSelectbox, div.stRadio, div.stMultiSelect {
    background-color: white;
    border-radius: 12px;
}

/* Restaurar el texto a su color original de Streamlit */
body, p, span, label, div {
    color: inherit;
}
</style>
""", unsafe_allow_html=True)


# Menú lateral de navegación
st.sidebar.title("🧪 Sección de Análisis")

opcion = st.sidebar.selectbox(
    "Selecciona una opción de análisis:",
    (
        "Página Principal",
        "DataFrame de Brasil Rio de Janeiro",
        "Análisis Univariado",
        "Modelado Predictivo",
        "Correlaciones",
        "Regresiones",
    )
)


if opcion == "Página Principal":
    st.markdown("<h1 style='text-align: center; color: #0c8100;'>Explora Airbnb en Brasil</h1>", unsafe_allow_html=True)
    st.markdown("### Bienvenido a un análisis interactivo y visual de los datos de Airbnb en Brasil.")
    st.markdown("---")

    st.markdown("""
    Explora los aspectos clave del mercado de alojamientos en ciudades brasileñas, desde precios hasta disponibilidad. 
    Este dashboard está diseñado para ayudarte a entender mejor cómo funciona la oferta de hospedajes en diferentes zonas turísticas.
    """)

    st.subheader("🏙️ Lugares destacados de Brasil, Rio de Janeiro:")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(
            "https://fthmb.tqn.com/I6wZC3bVDYtPmFex6fyX39H1ixo=/Christ_the_Redeemer-140514153-56a02f845f9b58eba4af490a.jpg",
            caption="Cristo Redentor", use_container_width=True)
    with col2:
        st.image(
            "https://a.cdn-hotels.com/gdcs/production45/d528/a71ce87b-cde6-4b96-b9b1-68d7cdc0387e.jpg",
            caption="Isla Angra Dos", use_container_width=True)
    with col3:
        st.image(
            "https://content.skyscnr.com/m/28b2f34b194d5ba1/original/GettyImages-457745659.jpg?resize=1800px:1800px&quality=100",
            caption="Pan de Azúcar", use_container_width=True)
    
    st.markdown("---")

    st.markdown("### ¿Qué encontrarás en este dashboard?")
    st.markdown(
        """
        <div style="
            background-color: #f0fdf4;
            border-left: 8px solid #10b981;
            padding: 20px;
            border-radius: 10px;
            font-size: 16px;
            line-height: 1.8;
            margin-bottom: 20px;
        ">
            <p>🔍 <b>Análisis detallado de alojamientos:</b> Explora datos sobre ubicación, precios, disponibilidad y más.</p>
            <p>📊 <b>Modelos de predicción:</b> Estimaciones inteligentes de precios y tendencias.</p>
            <p>📍 <b>Visualización por ciudades:</b> Interactúa con mapas y gráficos por ubicación.</p>
            <p>🧠 <b>Comparaciones de precios y tipos:</b> Compara distintos tipos de alojamiento de forma visual.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### 📊 Resumen General de Datos Airbnb en Brasil")

    st.markdown("""
    Los siguientes graficos te ayudan a entder como ser todo este analisis y te puedas familiarizar con lo que 
    encontraras en toda esta exploracion.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Tipos de alojamiento más comunes")
        st.caption("Explora qué tipo de alojamiento es más ofrecido por los anfitriones.")
        tipo_fig = px.histogram(df, x='room_type', color='room_type', title='Distribución por tipo de habitación')
        st.plotly_chart(tipo_fig, use_container_width=True)

    with col2:
        st.markdown("#### Barrios con más alojamientos")
        st.caption("Estos son los barrios con la mayor concentración de alojamientos en la base de datos.")
        top_barrios = df['neighbourhood_cleansed'].value_counts().head(5).reset_index()
        top_barrios.columns = ['Barrio', 'Cantidad']
        barrio_fig = px.bar(top_barrios, x='Barrio', y='Cantidad', title='Top 5 barrios con más alojamientos')
        st.plotly_chart(barrio_fig, use_container_width=True)

    st.markdown("---")


    st.subheader("🎥 Explora Brasil en video")
    videos = [
        {"titulo": "Cristo Redentor", "url": "https://youtu.be/bdMoU-rx10I?si=K4HtoxkZqz3LaIgg"},
        {"titulo": "Isla Angra Dos", "url": "https://youtu.be/-jAP8-pABxs?si=LceKdnIYz2URHMMW"},
        {"titulo": "Pan de Azúcar", "url": "https://youtu.be/7hRczs7H4go?si=OCr-bAsCuJUJahr8"}
    ]

    if "video_index" not in st.session_state:
        st.session_state.video_index = 0

    def anterior():
        st.session_state.video_index = (st.session_state.video_index - 1) % len(videos)

    def siguiente():
        st.session_state.video_index = (st.session_state.video_index + 1) % len(videos)

    video_actual = videos[st.session_state.video_index]
    st.markdown(f"### {video_actual['titulo']}")
    st.video(video_actual["url"])

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.button("⏮️ Anterior", on_click=anterior)
    with col3:
        st.button("Siguiente ⏭️", on_click=siguiente)

    st.markdown("""
        <h4>📍 Ubicación de Rio de Janeiro, Brasil </h4>
            <iframe 
                src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d470398.5431827502!2d-43.77564265922122!3d-22.91379063592589!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x9bde559108a05b%3A0x50dc426c672fd24e!2sR%C3%ADo%20de%20Janeiro%2C%20Estado%20de%20R%C3%ADo%20de%20Janeiro%2C%20Brasil!5e0!3m2!1ses-419!2smx!4v1746765036051!5m2!1ses-419!2smx"
                width="100%" 
                height="400" 
                style="border:0; border-radius: 12px;" 
                allowfullscreen="" 
                loading="lazy" 
                referrerpolicy="no-referrer-when-downgrade">
            </iframe>
            """, unsafe_allow_html=True)


st.markdown("---")

# 🟢 OPCIÓN: MOSTRAR DATAFRAME DE FORMA CREATIVA
if opcion == "DataFrame de Brasil Rio de Janeiro":
    st.title("📄 Vista del DataFrame - Airbnb Brasil (Río de Janeiro)")
    
    st.markdown("""
    ¡Aquí puedes explorar la base de datos completa usada en este análisis!  
    📌 **Filtra, ordena y examina** los registros de manera interactiva.
    """)
    
    # Mostrar resumen rápido
    st.markdown("### 🔍 Resumen general del conjunto de datos")
    st.write("Número de filas:", df.shape[0])
    st.write("Número de columnas:", df.shape[1])
    
    # Botón para mostrar los primeros registros
    with st.expander("📊 Ver primeras filas del DataFrame"):
        st.dataframe(df.head(20), use_container_width=True)

    # Selección de número de filas para visualizar
    st.markdown("### 📌 ¿Cuántas filas deseas visualizar?")
    num_filas = st.slider("Selecciona el número de filas a mostrar", 5, 100, 10)
    st.dataframe(df.head(num_filas), use_container_width=True)

    # Muestra columnas, tipos y nulos
    st.markdown("### 🧾 Información técnica del DataFrame")
    with st.expander("🧠 Tipos de datos y valores nulos"):
        info_df = pd.DataFrame({
            'Tipo de dato': df.dtypes,
            'Valores nulos': df.isnull().sum(),
            'Valores únicos': df.nunique()
        })
        st.dataframe(info_df)


# 🔵 OPCIÓN: ANÁLISIS UNIVARIADO
if opcion == "Análisis Univariado":
    st.title("📊 Análisis Univariado de Variables Categóricas")
    st.markdown("Selecciona una variable categórica para explorar su distribución:")

    categoria = st.selectbox("Variables categóricas disponibles:", text_cols)

    if categoria:
        st.markdown("### 📌 Distribución general")

        # Histograma (ya existente)
        fig1 = px.histogram(df, x=categoria, color=categoria, 
                            title=f"Distribución de: {categoria}", 
                            color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig1, use_container_width=True)

        # Gráfico de barras horizontal
        st.markdown("### 🔁 Gráfico de barras horizontal")
        freq = df[categoria].value_counts().reset_index()
        freq.columns = [categoria, 'Frecuencia']
        fig2 = px.bar(freq, x='Frecuencia', y=categoria, orientation='h',
                      color=categoria, title="Frecuencia de categorías",
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig2, use_container_width=True)

        # Gráfico circular (pie chart)
        st.markdown("### 🍩 Gráfico circular")
        fig3 = px.pie(freq, values='Frecuencia', names=categoria, 
                      title="Proporción de categorías", 
                      color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig3, use_container_width=True)

        # Tabla de frecuencias
        st.markdown("### 📋 Tabla de frecuencia de categorías")
        st.dataframe(freq)

#Modelo Predictivo
elif opcion == "Modelado Predictivo":
    st.title("📉 Modelado Predictivo")
    st.markdown("Explora relaciones entre variables con herramientas visuales y predice comportamientos.")
    st.subheader("📌 Dispersión entre Variables")

    # Selección automática sin botón
    x_var = st.selectbox("Variable X (independiente):", numeric_cols)
    y_var = st.selectbox("Variable Y (dependiente):", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)

    color_var = st.selectbox("Variable categórica para color (opcional):", text_cols)

    # Mostrar el scatterplot directamente
    fig_scatter = px.scatter(df, x=x_var, y=y_var, color=df[color_var],
                             title=f"Dispersión: {y_var} vs {x_var} coloreado por {color_var}",
                             color_discrete_sequence=px.colors.qualitative.Set1)
    st.plotly_chart(fig_scatter, use_container_width=True)


#CORRELACIONES
if opcion == "Correlaciones":
    st.title("🔗 Matriz de Correlaciones")
    st.markdown("Explora la relación entre variables numéricas con una matriz de correlación interactiva.")

    # Asegurar que numeric_cols sea lista
    if isinstance(numeric_cols, pd.Index):
        numeric_cols = list(numeric_cols)

    if not numeric_cols:
        st.warning("⚠️ No hay columnas numéricas disponibles para calcular correlaciones.")
    else:
        # Calcular matriz de correlación
        corr_matrix = df[numeric_cols].corr()

        # Mostrar como tabla
        st.subheader("📋 Tabla de Correlaciones")
        st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm').format("{:.2f}"))

        # Mostrar como gráfico de calor (heatmap)
        st.subheader("📊 Mapa de Calor de Correlaciones")
        fig = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale='RdBu_r',
            title="Heatmap de Correlaciones",
            width=1200,  # más ancho
            height=1000  # más alto
        )
        st.plotly_chart(fig, use_container_width=True)

        # Matriz de dispersión opcional
        st.subheader("🔍 Matriz de dispersión (solo si seleccionas pocas variables)")
        selected_corr_vars = st.multiselect("Selecciona variables para graficar matriz de dispersión:", numeric_cols)

        if selected_corr_vars and len(selected_corr_vars) <= 5:
            scatter_fig = px.scatter_matrix(
                df[selected_corr_vars],
                dimensions=selected_corr_vars,
                title="Matriz de Dispersión",
                color_discrete_sequence=px.colors.qualitative.Plotly  # más colores bonitos
            )
            st.plotly_chart(scatter_fig, use_container_width=True)
        elif selected_corr_vars:
            st.warning("⚠️ Por favor selecciona 5 variables o menos para la matriz de dispersión.")


#Regresiones 
elif opcion == "Regresiones":
    st.title("📈 Análisis de Regresiones")
    st.markdown("Selecciona el tipo de regresión que deseas aplicar:")

    tipo_regresion = st.selectbox("Tipo de regresión:", [
        "Regresión Lineal Simple",
        "Regresión Lineal Múltiple",
        "Regresión Logística"
    ])

    # 👉 REGRESIÓN LINEAL SIMPLE
    if tipo_regresion == "Regresión Lineal Simple":
        st.subheader("📊 Regresión Lineal Simple")

        x_var = st.selectbox("Selecciona la variable independiente (X):", numeric_cols)
        y_var = st.selectbox("Selecciona la variable dependiente (Y):", numeric_cols)

        if x_var and y_var:
            X = df[[x_var]]
            y = df[y_var]

            modelo = LinearRegression()
            modelo.fit(X, y)

            y_pred = modelo.predict(X)
            r2 = r2_score(y, y_pred)

            st.markdown(f"<h4>📌 Resultados:</h4>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:20px'><b>Coeficiente:</b> {modelo.coef_[0]:.4f}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:20px'><b>Intercepto:</b> {modelo.intercept_:.4f}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:20px'><b>R² Score:</b> {r2:.4f}</p>", unsafe_allow_html=True)

            # Gráfico
            fig = px.scatter(df, x=x_var, y=y_var, title="📉 Ajuste de la regresión lineal simple", labels={x_var: x_var.capitalize(), y_var: y_var.capitalize()})
            # Línea de regresión
            fig.add_scatter(x=df[x_var], y=y_pred, mode='lines', name='Línea de regresión', line=dict(color='green', width=2))
            # Predicciones
            fig.add_scatter(x=df[x_var], y=y_pred, mode='markers', name='Predicciones', marker=dict(color='red', size=6))
            # Datos reales
            fig.add_scatter(x=df[x_var], y=df[y_var], mode='markers', name='Datos Reales', marker=dict(color='blue', size=6))
            st.plotly_chart(fig, use_container_width=True)

    # 👉 REGRESIÓN LINEAL MÚLTIPLE
    if tipo_regresion == "Regresión Lineal Múltiple":
        st.subheader("📊 Regresión Lineal Múltiple")

        y_var = st.selectbox("Selecciona la variable dependiente:", numeric_cols)
        x_vars = st.multiselect("Selecciona variables independientes:", [col for col in numeric_cols if col != y_var])

        if len(x_vars) >= 1:
            X = df[x_vars]
            y = df[y_var]

            # Ajustar el modelo
            modelo = LinearRegression()
            modelo.fit(X, y)
            y_pred = modelo.predict(X)
            r2 = r2_score(y, y_pred)

            # Mostrar resultados
            st.markdown(f"<h4>📌 Resultados:</h4>", unsafe_allow_html=True)
            for i, col in enumerate(x_vars):
                st.markdown(f"<p style='font-size:18px'><b>{col}:</b> {modelo.coef_[i]:.4f}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:18px'><b>Intercepto:</b> {modelo.intercept_:.4f}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:18px'><b>R² Score:</b> {r2:.4f}</p>", unsafe_allow_html=True)

            # Crear el dataframe de coeficientes
            coef_df = pd.DataFrame({
                "Variable": x_vars,
                "Coeficiente": modelo.coef_
            }).sort_values(by="Coeficiente", key=abs, ascending=False)

            # Gráfico de importancia de variables
            fig = px.bar(coef_df, x="Variable", y="Coeficiente", 
                        title="📊 Importancia de Variables en la Predicción",
                        color="Coeficiente", 
                        color_continuous_scale="Viridis",
                        labels={"Coeficiente": "Impacto en la Predicción"})
            
            fig.update_layout(xaxis_title="Variables Independientes", yaxis_title="Coeficiente")
            st.plotly_chart(fig, use_container_width=True)



    # 👉 REGRESIÓN LOGÍSTICA
    elif tipo_regresion == "Regresión Logística":
        st.subheader("📊 Regresión Logística")

        cat_target = st.selectbox("Selecciona una variable categórica binaria como objetivo:", 
                                  [col for col in text_cols if df[col].nunique() == 2])

        x_vars = st.multiselect("Selecciona variables numéricas para entrenar:", numeric_cols)

        if cat_target and len(x_vars) > 0:
            df_model = df[[cat_target] + x_vars].dropna()

            X = df_model[x_vars]
            y = df_model[cat_target].astype('category').cat.codes  # Convertir a binario

            model = LogisticRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)[:, 1]

            acc = accuracy_score(y, y_pred)
            cm = confusion_matrix(y, y_pred)
            report = classification_report(y, y_pred, output_dict=True)
            fpr, tpr, _ = roc_curve(y, y_proba)
            roc_auc = roc_auc_score(y, y_proba)

            # Resultados visuales
            st.markdown(f"<h4>📌 Resultados:</h4>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:20px'><b>Exactitud (Accuracy):</b> {acc:.4f}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:20px'><b>AUC Score:</b> {roc_auc:.4f}</p>", unsafe_allow_html=True)

            # Gráfico: Matriz de confusión
            fig_cm = px.imshow(cm, text_auto=True, title="🔍 Matriz de Confusión", 
                               x=["Clase 0", "Clase 1"], y=["Clase 0", "Clase 1"],
                               color_continuous_scale='blues')
            st.plotly_chart(fig_cm)
