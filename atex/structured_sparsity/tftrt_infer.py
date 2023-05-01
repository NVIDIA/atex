# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.


import os
import copy
import argparse
import time

from statistics import mean

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import numpy as np
import tensorflow as tf

from tensorflow.python.compiler.tensorrt import trt_convert as trt

SAVEDMODEL_PATH = "exported_model"


def load_and_convert(path, precision):
    """ Load a saved model and convert it to FP32 or FP16. Return a converter """

    params = copy.deepcopy(trt.DEFAULT_TRT_CONVERSION_PARAMS)

    params = params._replace(
        precision_mode=(
            trt.TrtPrecisionMode.FP16
            if precision.lower() == "fp16" else
            trt.TrtPrecisionMode.FP32
        ),
        max_workspace_size_bytes=2 << 32,  # 8,589,934,592 bytes
        maximum_cached_engines=100,
        minimum_segment_size=3,
        allow_build_at_runtime=True,
    )

    import pprint
    print("%" * 85)
    pprint.pprint(params)
    print("%" * 85)

    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=path,
        conversion_params=params,
        use_dynamic_shape=False,
    )

    return converter


if __name__ == "__main__":

    BATCH_SIZE = 32  # This number will depend on the size of your dataset.

    INFERENCE_STEPS = 2000
    WARMUP_STEPS = 150

    parser = argparse.ArgumentParser(prog='mydaemon')

    feature_parser = parser.add_mutually_exclusive_group(required=True)
    feature_parser.add_argument('--use_native_tensorflow', dest="use_tftrt", help="help", action='store_false')
    feature_parser.add_argument('--use_tftrt_model', dest="use_tftrt", action='store_true')

    args = parser.parse_args()

    print("\n=========================================")
    print("Inference using: {} ...".format(
        "TF-TRT" if args.use_tftrt else "Native Tensorflow")
    )
    print("=========================================\n")
    time.sleep(2)

    def dataloader_fn(batch_size):
        dataset = tf.data.Dataset.from_tensor_slices(np.random.uniform(size=(1, 224, 224, 3)).astype(np.float32))
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.take(count=1)  # loop over 1 batch
        dataset = dataset.cache()
        dataset = dataset.repeat()
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

    if args.use_tftrt:
        converter = load_and_convert(
            os.path.join(SAVEDMODEL_PATH),
            precision="fp16",
        )
        xx = converter.convert()
        converter.save(
            os.path.join(SAVEDMODEL_PATH, "converted")
        )

        root = tf.saved_model.load(os.path.join(SAVEDMODEL_PATH, "converted"))
    else:
        root = tf.saved_model.load(SAVEDMODEL_PATH)

    infer_fx = root.signatures['serving_default']

    try:
        output_tensorname = list(infer_fx.structured_outputs.keys())[0]
    except AttributeError:
        # Output tensor doesn't have a name, index 0
        output_tensorname = 0

    ds = dataloader_fn(
        batch_size=BATCH_SIZE
    )
    iterator = iter(ds)

    @tf.function
    def infer_step(batch_x):
      return infer_fx(batch_x)[output_tensorname]

    try:
        step_times = list()
        for step in range(1, INFERENCE_STEPS + 1):
            input_batch = iterator.get_next()
            if step % 100 == 0:
                print("Processing step: %04d ..." % step)
            start_t = time.time()
            probs = infer_step(input_batch).numpy()
            step_time = time.time() - start_t
            if step >= WARMUP_STEPS:
                step_times.append(step_time)
    except tf.errors.OutOfRangeError:
        pass

    avg_step_time = mean(step_times)
    print("\nAverage step time: %.1f msec" % (avg_step_time * 1e3))
    print("Average throughput: %d samples/sec" % (
        BATCH_SIZE / avg_step_time
    ))
