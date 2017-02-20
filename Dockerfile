FROM missinglinkai/frameworks:latest

RUN python -m pip install missinglink-sdk

ADD keras_mnist.py keras_mnist.py
ADD keras_mnist_load_data.py keras_mnist_load_data.py

ENV PROJECT_TOKEN=HnHgRzkuOpVNTqMK
ENV OWNER_ID=ffff-cf7d-6501-e583-8c13a14eca0d

RUN python keras_mnist_load_data.py

CMD python keras_mnist.py \
    --owner-id $OWNER_ID \
    --project-token $PROJECT_TOKEN \
    --epochs 10 \
    --host https://missinglink-staging.appspot.com
