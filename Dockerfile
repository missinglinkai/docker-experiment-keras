FROM missinglinkai/frameworks:latest

RUN python -m pip install missinglink-sdk

ADD keras_mnist.py keras_mnist.py

RUN python keras_mnist.py \
    --owner-id 381d23e4-d368-508f-f19b-48c3d8420c60 \
    --project-token YCbtEryxyosBKYgx \
    --epochs 0 \
    --host https://missinglink-staging.appspot.com

CMD python keras_mnist.py \
    --owner-id 381d23e4-d368-508f-f19b-48c3d8420c60 \
    --project-token YCbtEryxyosBKYgx \
    --epochs 2 \
    --host https://missinglink-staging.appspot.com