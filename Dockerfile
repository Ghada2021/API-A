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

COPY dataset.csv ./dataset.csv

COPY Model.py ./Model.py


RUN python3 Model.py