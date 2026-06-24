import onnxruntime as ort
import onnxruntime_qnn as qnn_ep

print("QNN library path:", qnn_ep.get_library_path())

ort.register_execution_provider_library("QNNExecutionProvider", qnn_ep.get_library_path())

devices = ort.get_ep_devices()
for d in devices:
    print(d.ep_name, "-", d)
