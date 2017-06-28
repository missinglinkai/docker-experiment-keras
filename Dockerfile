FROM missinglinkai/frameworks:latest

ADD keras_mnist.py keras_mnist.py
ADD keras_mnist_load_data.py keras_mnist_load_data.py

RUN python keras_mnist_load_data.py

RUN python -m pip install missinglink-sdk -U

ENV PROJECT_TOKEN=HnHgRzkuOpVNTqMK
ENV OWNER_ID=ffff-cf7d-6501-e583-8c13a14eca0d
ENV HOST=https://missinglink-staging.appspot.com
ENV EPOCHS=10
ENV BATCH_SIZE=128

CMD python keras_mnist.py \
    --owner-id $OWNER_ID \
    --project-token $PROJECT_TOKEN \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --host $HOST
