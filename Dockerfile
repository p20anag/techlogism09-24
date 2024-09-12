FROM python:3.12.3

ADD 123123123.py .
EXPOSE 8080

RUN pip install scikit-learn
RUN pip install pandas
RUN pip install streamlit
RUN pip install matplotlib

# RUN [ "python", "./123123123.py" ]
WORKDIR /app
ENTRYPOINT ["streamlit", "run", "123123123.py", "--server.port=8080", "--server.address=0.0.0.0"]