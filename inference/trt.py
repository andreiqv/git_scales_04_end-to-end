import tensorflow as tf
import uff
import tensorrt as trt
from tensorrt.parsers import uffparser

use_hub_model = True:
if use_hub_model:
	FROZEN_FPATH = '/home/andrei/Data/Datasets/Scales/pb/output_graph.pb'
	ENGINE_FPATH = '/home/andrei/Data/Datasets/Scales/pb/hub_model_engine.plan'
	INPUT_NODE = 'Placeholder-x'
	OUTPUT_NODE = 'final_result'
	INPUT_SIZE = [3, 299, 299]
else:
	FROZEN_FPATH = '/root/tmp/saved_model_inception_resnet.pb'
	ENGINE_FPATH = '/root/tmp/engine.plan'
	INPUT_NODE = 'Placeholder-x'
	OUTPUT_NODE = 'sigmoid_out'
	INPUT_SIZE = [3, 299, 299]

MAX_BATCH_SIZE = 1
MAX_WORKSPACE = 1 << 20

# convert TF frozen graph to UFF graph
uff_model = uff.from_tensorflow_frozen_model(FROZEN_FPATH, [OUTPUT_NODE])

# create UFF parser and logger
parser = uffparser.create_uff_parser()
parser.register_input(INPUT_NODE, INPUT_SIZE, 0)
parser.register_output(OUTPUT_NODE)
G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)

# Build optimized inference engine
engine = trt.utils.uff_to_trt_engine(G_LOGGER, uff_model, parser, MAX_BATCH_SIZE, MAX_WORKSPACE)

# Save inference engine
trt.utils.write_engine_to_file(ENGINE_FPATH, engine.serialize())

# Cleaning Up
parser.destroy()
engine.destroy()


"""

engine = trt.utils.uff_to_trt_engine(G_LOGGER, uff_model, parser, MAX_BATCH_SIZE, MAX_WORKSPACE)
[TensorRT] ERROR: UFFParser: Validator error: input/BottleneckInputPlaceholder: Unsupported operation _PlaceholderWithDefault
[TensorRT] ERROR: Failed to parse UFF model stream
  File "/usr/lib/python3.5/dist-packages/tensorrt/utils/_utils.py", line 255, in uff_to_trt_engine
    assert(parser.parse(stream, network, model_datatype))
Traceback (most recent call last):
  File "/usr/lib/python3.5/dist-packages/tensorrt/utils/_utils.py", line 255, in uff_to_trt_engine
    assert(parser.parse(stream, network, model_datatype))
AssertionError

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/lib/python3.5/dist-packages/tensorrt/utils/_utils.py", line 263, in uff_to_trt_engine
    raise AssertionError('UFF parsing failed on line {} in statement {}'.format(line, text))
AssertionError: UFF parsing failed on line 255 in statement assert(parser.parse(stream, network, model_datatype))
>>> 

"""