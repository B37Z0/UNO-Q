## Epic QNN (Execution Provider) Fail

**GPU Acceleration Check - UNO Q (Adreno 702)**

Qualcomm AI Engine Direct SDK (QNN) References:
- https://docs.ultralytics.com/integrations/qnn#measured-performance
- https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk

The Arduino UNO Q has two compute units worth benchmarking: the CPU and the Adreno 702 GPU. CPU inference has been verified (68ms average on MobileNetV2 INT8). An attempt was made to run and inference on the GPU.

`onnxruntime-qnn`
The QNN SDK enables AI acceleration across Snapdrawon hardware, including the Adreno GPU. However, instead of using the QNN SDK directly, `onnx-runtime-qnn` provides an easier route in directly using ONNX model exports (note the QNN execution provider requires fixed-shape input).
The full Qualcomm QNN SDK (via QPM) was ruled out in favor of `onnxruntime-qnn`, which is much lighter and can be installed directly in the board's Python environment. The GPU backend doesn't specifically require models to be quantized, so plain MobileNetV2 can be exported.


MobileNetV2 was exported to the ONNX format on the Windows laptop, then copied onto the board. When attempting to run solely on the GPU (CPU fallback disabled):
```
QNN_BACKEND_ERROR_CANNOT_INITIALIZE: Backend failed to initialize
```
This is the GPU backend itself refusing to start, which appears to be a compatibility wall.

**Why did it fail?**
It seems there are two different OpenCL runtimes for this Adreno GPU chip:
1. **Qualcomm's own driver** - proprietary, made by Qualcomm, shipped on phones and official Qualcomm dev kits.
2. **Mesa's open-source driver** (called Freedreno/rusticl) - built by an independent open-source community, reverse-engineered without Qualcomm's involvement. Shipped on the Arduino UNO Q's Linux image.

Both work for general graphics/compute tasks (tested and confirmed), but Qualcomm's GPU-accelerated AI toolkit was only built and tested against *Qualcomm's own* driver. Attempting to use the open-source driver fails.
- See Qualcomm's announcement of the **ONNX Runtime Qualcomm® AI Engine Direct (QNN) EP with the Qualcomm Adreno GPU backend** for implementation details. 

Presumably, **GPU-accelerated inference via QNN is not currently achievable on the Arduino UNO Q's stock software image.** See sources:
- https://www.qualcomm.com/developer/blog/2025/05/unlocking-power-of-qualcomm-qnn-execution-provider-gpu-backend-onnx-runtime - Qualcomm announcement
- https://mysupport.qualcomm.com/supportforums/s/question/0D5dK00000Q3WZcSAN/trouble-with-ai-inference-with-adreno-702-on-arduino-uno-q-possibly-missing-proprietary-qualcomm-opencl-driver - personal forum post
