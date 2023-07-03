FROM python:3.10

RUN pip install numpy --no-cache-dir
RUN pip install scipy --no-cache-dir
RUN pip install scikit-learn --no-cache-dir
RUN pip install pandas --no-cache-dir
RUN pip install Keras --no-cache-dir
RUN pip install lightgbm --no-cache-dir
RUN pip install matplotlib --no-cache-dir
RUN pip install xgboost --no-cache-dir
RUN pip install seaborn --no-cache-dir
RUN pip install joblib 
WORKDIR /home/jovyan 
RUN mkdir model raw_data processed_data results

ENV RAW_DATA_DIR=/home/jovyan/raw_data
ENV PROCESSED_DATA_DIR=/home/jovyan/processed_data
ENV MODEL_DIR=/home/jovyan/model
ENV RESULTS_DIR=/home/jovyan/results
ENV RAW_DATA_FILE=dataset.csv

COPY dataset.csv ./dataset.csv
COPY Preprocessing.py ./Preprocessing.py
COPY Model.py ./Model.py

CMD python3 Preprocessing.py
CMD python3 Model.py
