import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import json

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
Вы можете ввести данные одного пациента вручную или загрузить файл с данными нескольких пациентов.
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

# Выбор режима ввода
input_mode = st.radio(
    "Выберите способ ввода данных:",
    ["📝 Ручной ввод", "📁 Загрузка файла"],
    horizontal=True
)

# Функция для проверки и подготовки данных
def validate_and_prepare_data(df):
    required_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
                       'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
    
    # Проверяем наличие всех колонок
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return None, f"❌ В файле отсутствуют колонки: {', '.join(missing_columns)}"
    
    # Проверяем типы данных
    errors = []
    try:
        df['Age'] = pd.to_numeric(df['Age'])
        df['RestingBP'] = pd.to_numeric(df['RestingBP'])
        df['Cholesterol'] = pd.to_numeric(df['Cholesterol'])
        df['FastingBS'] = pd.to_numeric(df['FastingBS'])
        df['MaxHR'] = pd.to_numeric(df['MaxHR'])
        df['Oldpeak'] = pd.to_numeric(df['Oldpeak'])
    except Exception as e:
        errors.append(f"Ошибка в числовых данных: {e}")
    
    # Проверяем допустимые значения
    valid_sex = {'M', 'F'}
    valid_chest = {'ATA', 'NAP', 'ASY', 'TA'}
    valid_ecg = {'Normal', 'ST', 'LVH'}
    valid_angina = {'Y', 'N'}
    valid_slope = {'Up', 'Flat', 'Down'}
    
    invalid_sex = df[~df['Sex'].isin(valid_sex)]
    if len(invalid_sex) > 0:
        errors.append(f"Недопустимые значения в колонке Sex: {invalid_sex['Sex'].unique().tolist()}")
    
    invalid_chest = df[~df['ChestPainType'].isin(valid_chest)]
    if len(invalid_chest) > 0:
        errors.append(f"Недопустимые значения в колонке ChestPainType: {invalid_chest['ChestPainType'].unique().tolist()}")
    
    if errors:
        return None, "\n".join(errors)
    
    return df, None

# Функция для предсказания и отображения результатов
def make_predictions(df, df_original):
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)
    
    # Добавляем результаты
    df_results = df_original.copy()
    df_results['Prediction'] = predictions
    df_results['Risk_Probability'] = probabilities[:, 1]
    df_results['Risk_Level'] = df_results['Risk_Probability'].apply(
        lambda x: '🔴 Высокий' if x >= 0.7 else ('🟡 Средний' if x >= 0.3 else '🟢 Низкий')
    )
    df_results['Recommendation'] = df_results['Prediction'].apply(
        lambda x: '⚠️ Обратиться к кардиологу' if x == 1 else '✅ Профилактическое наблюдение'
    )
    
    return df_results

# Функция для отображения результатов
def display_results(df_results):
    # Статистика
    st.markdown("---")
    st.header("📊 Результаты предсказаний")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Всего пациентов", len(df_results))
    with col2:
        high_risk = (df_results['Prediction'] == 1).sum()
        st.metric("⚠️ Высокий риск", high_risk, 
                 delta=f"{(high_risk/len(df_results)*100):.1f}%")
    with col3:
        low_risk = (df_results['Prediction'] == 0).sum()
        st.metric("✅ Низкий риск", low_risk,
                 delta=f"{(low_risk/len(df_results)*100):.1f}%")
    with col4:
        avg_risk = df_results['Risk_Probability'].mean()
        st.metric("📊 Средний риск", f"{avg_risk:.1%}")
    
    # Таблица с результатами
    st.subheader("📋 Детальные результаты")
    
    display_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
                      'Prediction', 'Risk_Probability', 'Risk_Level', 'Recommendation']
    
    # Форматируем для отображения
    styled_df = df_results[display_columns].style.format({
        'Risk_Probability': '{:.1%}'
    }).background_gradient(
        subset=['Risk_Probability'],
        cmap='RdYlGn_r',
        vmin=0,
        vmax=1
    )
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Визуализация
    st.subheader("📊 Распределение уровней риска")
    risk_counts = df_results['Risk_Level'].value_counts()
    st.bar_chart(risk_counts)
    
    # Экспорт результатов
    col_export1, col_export2, col_export3 = st.columns(3)
    
    with col_export1:
        csv_results = df_results.to_csv(index=False)
        st.download_button(
            label="📥 Скачать CSV",
            data=csv_results,
            file_name="predictions_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_export2:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df_results.to_excel(writer, index=False, sheet_name='Predictions')
        excel_data = excel_buffer.getvalue()
        st.download_button(
            label="📊 Скачать Excel",
            data=excel_data,
            file_name="predictions_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col_export3:
        json_results = df_results.to_json(orient='records', indent=2)
        st.download_button(
            label="📄 Скачать JSON",
            data=json_results,
            file_name="predictions_results.json",
            mime="application/json",
            use_container_width=True
        )

if input_mode == "📝 Ручной ввод":
    # Существующая форма ручного ввода (оставляем без изменений)
    st.header("📋 Введите данные пациента")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Основные показатели")
        age = st.number_input("Возраст (лет)", min_value=0, max_value=120, value=50, step=1)
        sex = st.selectbox("Пол", options=['M', 'F'], format_func=lambda x: "Мужской" if x == 'M' else "Женский")
        chest_pain_type = st.selectbox(
            "Тип боли в груди",
            options=['ATA', 'NAP', 'ASY', 'TA'],
            format_func=lambda x: {
                'ATA': 'Атипичная стенокардия',
                'NAP': 'Неангинальная боль',
                'ASY': 'Бессимптомная',
                'TA': 'Типичная стенокардия'
            }[x]
        )
        resting_bp = st.number_input("Систолическое давление (мм рт. ст.)", min_value=0, max_value=300, value=120, step=5)
        cholesterol = st.number_input("Холестерин (мг/дл)", min_value=0, max_value=600, value=200, step=10)
    
    with col2:
        st.subheader("Электрокардиограмма и симптомы")
        fasting_bs = st.selectbox("Глюкоза натощак (>120 мг/дл)", options=[0, 1], 
                                 format_func=lambda x: "Нет (≤120 мг/дл)" if x == 0 else "Да (>120 мг/дл)")
        resting_ecg = st.selectbox(
            "Результаты ЭКГ в покое",
            options=['Normal', 'ST', 'LVH'],
            format_func=lambda x: {
                'Normal': 'Нормальный',
                'ST': 'Аномалии ST-T',
                'LVH': 'Гипертрофия левого желудочка'
            }[x]
        )
        max_hr = st.number_input("Максимальная частота пульса", min_value=0, max_value=250, value=150, step=5)
        exercise_angina = st.selectbox("Стенокардия при нагрузке", options=['Y', 'N'], 
                                      format_func=lambda x: "Да" if x == 'Y' else "Нет")
    
    col3, col4 = st.columns(2)
    
    with col3:
        oldpeak = st.number_input("Депрессия ST (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.1f")
    
    with col4:
        st_slope = st.selectbox(
            "Наклон ST сегмента",
            options=['Up', 'Flat', 'Down'],
            format_func=lambda x: {
                'Up': 'Подъем',
                'Flat': 'Плоский',
                'Down': 'Спуск'
            }[x]
        )
    
    predict_button = st.button("🔍 Получить предсказание", type="primary", use_container_width=True)
    
    if predict_button and model is not None:
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
        
        with st.spinner("Анализ данных..."):
            try:
                prediction = model.predict(input_data)[0]
                prediction_proba = model.predict_proba(input_data)[0]
                
                st.markdown("---")
                st.header("📊 Результат предсказания")
                
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
                    st.markdown("#### Вероятности:")
                    col_risk, col_safe = st.columns(2)
                    with col_risk:
                        st.metric("Риск заболевания", f"{prediction_proba[1]:.1%}")
                    with col_safe:
                        st.metric("Низкий риск", f"{prediction_proba[0]:.1%}")
                    
                    risk_level = prediction_proba[1]
                    st.progress(risk_level, text=f"Уровень риска: {risk_level:.1%}")
                    
            except Exception as e:
                st.error(f"Ошибка при предсказании: {e}")

elif input_mode == "📁 Загрузка файла":
    st.header("📁 Загрузка файла с данными")
    
    st.markdown("""
    ### Поддерживаемые форматы:
    - **CSV** (Comma-Separated Values)
    - **Excel** (.xlsx, .xls)
    - **JSON** (.json)
    - **Parquet** (.parquet)
    - **Текст** (.txt с разделителями)
    """)
    
    # Выбор формата файла
    file_format = st.selectbox(
        "Выберите формат файла",
        ["CSV", "Excel", "JSON", "Parquet", "Текст (Tab-delimited)"]
    )
    
    # Скачать шаблон
    if st.button("📥 Скачать шаблон данных"):
        template = pd.DataFrame({
            'Age': [55, 45, 60],
            'Sex': ['M', 'F', 'M'],
            'ChestPainType': ['ATA', 'NAP', 'ASY'],
            'RestingBP': [140, 120, 150],
            'Cholesterol': [250, 200, 300],
            'FastingBS': [0, 0, 1],
            'RestingECG': ['Normal', 'Normal', 'ST'],
            'MaxHR': [150, 160, 140],
            'ExerciseAngina': ['N', 'N', 'Y'],
            'Oldpeak': [1.0, 0.5, 2.0],
            'ST_Slope': ['Flat', 'Up', 'Down']
        })
        
        # Предлагаем разные форматы для скачивания
        col_download1, col_download2 = st.columns(2)
        
        with col_download1:
            csv = template.to_csv(index=False)
            st.download_button(
                label="📥 CSV шаблон",
                data=csv,
                file_name="template_heart_disease.csv",
                mime="text/csv"
            )
        
        with col_download2:
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                template.to_excel(writer, index=False, sheet_name='Template')
            excel_data = excel_buffer.getvalue()
            st.download_button(
                label="📊 Excel шаблон",
                data=excel_data,
                file_name="template_heart_disease.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    # Загрузка файла
    uploaded_file = st.file_uploader(
        "Выберите файл для загрузки",
        type=['csv', 'xlsx', 'xls', 'json', 'parquet', 'txt'],
        help="Поддерживаются форматы: CSV, Excel, JSON, Parquet, TXT"
    )
    
    if uploaded_file is not None and model is not None:
        try:
            # Читаем файл в зависимости от формата
            if file_format == "CSV" or uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif file_format == "Excel" or uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            elif file_format == "JSON" or uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            elif file_format == "Parquet" or uploaded_file.name.endswith('.parquet'):
                df = pd.read_parquet(uploaded_file)
            elif file_format == "Текст (Tab-delimited)" or uploaded_file.name.endswith('.txt'):
                df = pd.read_csv(uploaded_file, sep='\t')
            else:
                # Автоопределение
                try:
                    df = pd.read_csv(uploaded_file)
                except:
                    try:
                        df = pd.read_excel(uploaded_file)
                    except:
                        df = pd.read_json(uploaded_file)
            
            st.success(f"✅ Файл успешно загружен! Найдено {len(df)} записей")
            
            # Показываем предпросмотр
            with st.expander("📊 Предпросмотр данных"):
                st.dataframe(df.head(10))
                st.write(f"Всего записей: {len(df)}")
                st.write(f"Колонки: {', '.join(df.columns)}")
            
            # Валидация данных
            df_validated, error_msg = validate_and_prepare_data(df)
            
            if error_msg:
                st.error(error_msg)
                st.info("Пожалуйста, используйте шаблон для загрузки данных или проверьте формат файла")
            else:
                if st.button("🔍 Выполнить предсказание", type="primary", use_container_width=True):
                    with st.spinner(f"Анализ {len(df_validated)} записей..."):
                        try:
                            # Делаем предсказания
                            df_results = make_predictions(df_validated, df)
                            display_results(df_results)
                            
                        except Exception as e:
                            st.error(f"Ошибка при предсказании: {e}")
        
        except Exception as e:
            st.error(f"Ошибка при чтении файла: {e}")
            st.info("Проверьте, что файл имеет правильный формат и структуру данных")