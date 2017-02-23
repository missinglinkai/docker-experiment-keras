FROM missinglinkai/frameworks:latest

RUN python -m pip install -i https://testpypi.python.org/pypi missinglink-sdk

ADD keras_mnist.py keras_mnist.py
ADD keras_mnist_load_data.py keras_mnist_load_data.py

ENV PROJECT_TOKEN=HnHgRzkuOpVNTqMK
ENV OWNER_ID=ffff-cf7d-6501-e583-8c13a14eca0d
ENV HOST=https://missinglink-staging.appspot.com
ENV EPOCHS=10

RUN python keras_mnist_load_data.py

CMD python keras_mnist.py \
    --owner-id $OWNER_ID \
    --project-token $PROJECT_TOKEN \
    --epochs $EPOCHS \
    --host $HOST
