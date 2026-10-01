FROM apache/airflow:3.3.2

# 1) 운영체제 라이브러리는 root 권한으로 설치
USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# 2) Python 패키지는 airflow 사용자로 설치 (Airflow 버전이 바뀌지 않도록 고정)
USER airflow
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" \
    lightgbm scikit-learn python-dotenv joblib