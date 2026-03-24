import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
import sklearn
sklearn.set_config(transform_output="pandas")


# Настройка страницы
st.set_page_config(
    page_title="Предсказание болезни сердца",
    page_icon="❤️",
    layout="wide"
)

# Заголовок
st.title("❤️ Предсказание сердечно-сосудистых заболеваний")
st.markdown("""
Это приложение использует машинное обучение для оценки риска развития болезни сердца.
Пожалуйста, введите данные пациента ниже для получения предсказания.
""")

# Загрузка модели
@st.cache_resource
def load_model():
    try:
        model = joblib.load('final_pipeline.pkl')
        return model
    except FileNotFoundError:
        st.error("Модель не найдена! Пожалуйста, убедитесь, что файл 'final_pipeline.pkl' находится в той же директории.")
        return None
    except Exception as e:
        st.error(f"Ошибка при загрузке модели: {e}")
        return None

model = load_model()

if model is not None:
    # Создание формы ввода данных
    st.header("📋 Введите данные пациента")
    
    # Разделение на колонки для лучшего расположения
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Основные показатели")
        age = st.number_input(
            "Возраст (лет)",
            min_value=0,
            max_value=120,
            value=50,
            step=1,
            help="Возраст пациента в годах"
        )
        
        sex = st.selectbox(
            "Пол",
            options=['M', 'F'],
            format_func=lambda x: "Мужской" if x == 'M' else "Женский",
            help="Пол пациента"
        )
        
        chest_pain_type = st.selectbox(
            "Тип боли в груди",
            options=['ATA', 'NAP', 'ASY', 'TA'],
            format_func=lambda x: {
                'ATA': 'Атипичная стенокардия',
                'NAP': 'Неангинальная боль',
                'ASY': 'Бессимптомная',
                'TA': 'Типичная стенокардия'
            }[x],
            help="Тип боли в груди"
        )
        
        resting_bp = st.number_input(
            "Систолическое артериальное давление (мм рт. ст.)",
            min_value=0,
            max_value=300,
            value=120,
            step=5,
            help="Давление в состоянии покоя"
        )
        
        cholesterol = st.number_input(
            "Холестерин (мг/дл)",
            min_value=0,
            max_value=600,
            value=200,
            step=10,
            help="Уровень холестерина в крови"
        )
    
    with col2:
        st.subheader("Электрокардиограмма и симптомы")
        fasting_bs = st.selectbox(
            "Уровень глюкозы натощак (>120 мг/дл)",
            options=[0, 1],
            format_func=lambda x: "Нет (≤120 мг/дл)" if x == 0 else "Да (>120 мг/дл)",
            help="Превышает ли уровень глюкозы натощак 120 мг/дл"
        )
        
        resting_ecg = st.selectbox(
            "Результаты ЭКГ в покое",
            options=['Normal', 'ST', 'LVH'],
            format_func=lambda x: {
                'Normal': 'Нормальный',
                'ST': 'Аномалии ST-T',
                'LVH': 'Гипертрофия левого желудочка'
            }[x],
            help="Результаты электрокардиограммы в покое"
        )
        
        max_hr = st.number_input(
            "Максимальная частота пульса",
            min_value=0,
            max_value=250,
            value=150,
            step=5,
            help="Максимальная достигнутая частота пульса"
        )
        
        exercise_angina = st.selectbox(
            "Стенокардия при физической нагрузке",
            options=['Y', 'N'],
            format_func=lambda x: "Да" if x == 'Y' else "Нет",
            help="Возникает ли боль в груди при физической нагрузке"
        )
    
    # Третья колонка для дополнительных параметров
    col3, col4 = st.columns(2)
    
    with col3:
        oldpeak = st.number_input(
            "Депрессия ST (Oldpeak)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
            format="%.1f",
            help="Депрессия ST, вызванная нагрузкой"
        )
    
    with col4:
        st_slope = st.selectbox(
            "Наклон ST сегмента",
            options=['Up', 'Flat', 'Down'],
            format_func=lambda x: {
                'Up': 'Подъем',
                'Flat': 'Плоский',
                'Down': 'Спуск'
            }[x],
            help="Наклон сегмента ST при пиковой нагрузке"
        )
    
    # Кнопка предсказания
    st.markdown("---")
    predict_button = st.button("🔍 Получить предсказание", type="primary", use_container_width=True)
    
    if predict_button:
        # Создание DataFrame из введенных данных
        input_data = pd.DataFrame({
            'Age': [age],
            'Sex': [sex],
            'ChestPainType': [chest_pain_type],
            'RestingBP': [resting_bp],
            'Cholesterol': [cholesterol],
            'FastingBS': [fasting_bs],
            'RestingECG': [resting_ecg],
            'MaxHR': [max_hr],
            'ExerciseAngina': [exercise_angina],
            'Oldpeak': [oldpeak],
            'ST_Slope': [st_slope]
        })
        
        # Показываем спиннер во время предсказания
        with st.spinner("Анализ данных..."):
            try:
                # Получаем предсказание и вероятности
                prediction = model.predict(input_data)[0]
                prediction_proba = model.predict_proba(input_data)[0]
                
                # Отображение результатов
                st.markdown("---")
                st.header("📊 Результат предсказания")
                
                # Создаем колонки для результатов
                result_col1, result_col2 = st.columns(2)
                
                with result_col1:
                    if prediction == 1:
                        st.error("### ⚠️ Высокий риск сердечного заболевания")
                        st.markdown("""
                        **Рекомендации:**
                        - Обратитесь к кардиологу для детального обследования
                        - Следите за артериальным давлением
                        - Поддерживайте здоровый образ жизни
                        - Регулярно проходите медицинские осмотры
                        """)
                    else:
                        st.success("### ✅ Низкий риск сердечного заболевания")
                        st.markdown("""
                        **Рекомендации:**
                        - Продолжайте вести здоровый образ жизни
                        - Регулярно проходите профилактические осмотры
                        - Поддерживайте физическую активность
                        - Следите за питанием
                        """)
                
                with result_col2:
                    # Отображение вероятностей
                    st.markdown("#### Вероятности:")
                    col_risk, col_safe = st.columns(2)
                    with col_risk:
                        st.metric(
                            "Риск заболевания",
                            f"{prediction_proba[1]:.1%}",
                            delta=None
                        )
                    with col_safe:
                        st.metric(
                            "Низкий риск",
                            f"{prediction_proba[0]:.1%}",
                            delta=None
                        )
                    
                    # Прогресс-бар для визуализации
                    risk_level = prediction_proba[1]
                    st.progress(risk_level, text=f"Уровень риска: {risk_level:.1%}")
                
                # Дополнительная информация
                with st.expander("ℹ️ О чем говорят эти результаты"):
                    st.markdown("""
                    **Интерпретация результатов:**
                    - **Низкий риск (0-30%)**: Вероятность заболевания низкая, но важно поддерживать здоровый образ жизни
                    - **Умеренный риск (30-70%)**: Требуется внимание к факторам риска, рекомендуется консультация врача
                    - **Высокий риск (70-100%)**: Высокая вероятность заболевания, необходима срочная консультация кардиолога
                    
                    **Важно:** Это предсказание основано на модели машинного обучения и не заменяет профессиональную медицинскую консультацию.
                    """)
                    
            except Exception as e:
                st.error(f"Ошибка при предсказании: {e}")
                st.info("Пожалуйста, проверьте введенные данные и попробуйте снова.")
    
    # Боковая панель с информацией
    with st.sidebar:
        st.header("📌 Информация о модели")
        st.markdown("""
        **Модель:** Voting Classifier (Ансамбль моделей)
        
        **Используемые алгоритмы:**
        - Random Forest
        - CatBoost
        - XGBoost
        - K-Nearest Neighbors
        
        **Метрики качества:**
        - Accuracy: ~92%
        - Кросс-валидация: 5-fold KFold
        
        **Примечание:** Модель обучена на данных о сердечно-сосудистых заболеваниях.
        """)
        
        st.header("📊 Описание признаков")
        with st.expander("Клинические показатели"):
            st.markdown("""
            - **Age**: Возраст в годах
            - **Sex**: Пол (M - мужской, F - женский)
            - **ChestPainType**: Тип боли в груди
            - **RestingBP**: Систолическое давление в покое (мм рт. ст.)
            - **Cholesterol**: Уровень холестерина (мг/дл)
            - **FastingBS**: Глюкоза натощак > 120 мг/дл (1 - да, 0 - нет)
            - **RestingECG**: Результаты ЭКГ в покое
            - **MaxHR**: Максимальная частота пульса
            - **ExerciseAngina**: Стенокардия при нагрузке (Y - да, N - нет)
            - **Oldpeak**: Депрессия ST
            - **ST_Slope**: Наклон ST сегмента
            """)
else:
    st.error("Не удалось загрузить модель. Пожалуйста, проверьте наличие файла 'final_pipeline.pkl'.")