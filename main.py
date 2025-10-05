import numpy as np
import pandas as pd
import feature_selection as ftr

if __name__=='__main__':
    age = []
    sleepDur = []
    sleepQuality = []
    stressLevel = []
    job_list = []
    physicalAct = []
    gender = []
    bmi = []
    systolic = []
    diastolic = []
    heartrate = []
    steps = []
    sleepDisorder = []

    file_path = "Feature-Selection-on-Sleep-data/updated_data_2.csv"  
    ftr.read_data(file_path, gender, age, job_list, sleepDur, sleepQuality, physicalAct,
            stressLevel, bmi, systolic, diastolic, heartrate, steps, sleepDisorder)

    ftr.spearman_with_ties(stressLevel, sleepQuality, ['stressLevel', 'sleepQuality'])
    ftr.pearson(stressLevel, sleepQuality, ['stressLevel', 'sleepQuality'])

    # ftures = [job_list, sleepDur, sleepQuality, physicalAct, age]
    # for var in ftures:
    #     x = np.array(var).reshape(-1, 1)
    #     y = np.array(sleepQuality)
    #     plot_regression_with_density(x, y)
    X = np.column_stack([job_list, sleepDur, sleepQuality, physicalAct, age])  # Shape: (8, 3)
    y = stressLevel  # Shape: (8,)
    ftr.plot_regres_feature(X, y)

    ftr.plot_polyreg_SVR(X, y)
    feature_names = ['job_activity', 'sleep_duration', 'Sleep Quality', 'physical_activity', 'age']
    ftr.plot_svr_scatter_with_lines(X, y, feature_names, kernel='rbf', C=1.0)

    ftr.plot_spline_importance(X, y, feature_names)
    ftr.plot_spline_with_lines(X, y, feature_names)


    data = pd.DataFrame({
        'gender': gender,
        'age': age,  # Y = X^2
        'Job': job_list,
        'Sleep Duration': sleepDur,
        'Quality of sleep': sleepQuality,
        'Physical Activity': physicalAct,
        'Stress Level': stressLevel,
        'BMI': bmi,
        'Heart Rate': heartrate,
        'Daily Steps': steps,
        'Sleep Disorder': sleepDisorder,
        'Systolic': systolic,
        'Diastolic': diastolic
    })


    # построение корреляционной матрицы
    ftr.plot_cov_matrix(data)
    # построение корреляционной матрицы

    # определения закона распределения
    ftr.get_law(stressLevel)
    # определения закона распределения
