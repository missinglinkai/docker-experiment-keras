FROM gw000/keras:1.2.2-cpu

workdir /experiment
ADD requirements.txt requirements.txt
ADD keras_mnist.py keras_mnist.py
ADD custom_metrics.py custom_metrics.py
ADD keras_mnist_load_data.py keras_mnist_load_data.py
ADD .git .git
RUN python keras_mnist_load_data.py
RUN sed -i s,'https://github.com/missinglinkai/docker-experiment-keras.git','git@github.com:missinglinkai/docker-experiment-keras.git',g  .git/config
RUN python -m pip install -r requirements.txt

ENV PROJECT_TOKEN=HnHgRzkuOpVNTqMK
ENV OWNER_ID=ffff-cf7d-6501-e583-8c13a14eca0d
ENV HOST=https://missinglink-staging.appspot.com
ENV EPOCHS=10
ENV BATCH_SIZE=128
ENV IS_SAMPLING=False

CMD python keras_mnist.py \
    --owner-id $OWNER_ID \
    --project-token $PROJECT_TOKEN \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --host $HOST \
    --is-sampling $IS_SAMPLING
