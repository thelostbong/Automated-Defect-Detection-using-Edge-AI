import onnx
import onnxruntime as ort
import numpy as np

MODEL_PATH = "best.onnx"  # Replace with your actual filename

try:
    # 1. Load the model and check structure
    onnx_model = onnx.load(MODEL_PATH)
    onnx.checker.check_model(onnx_model)
    print("✅ Model structure is valid (Schema check passed).")

    # 2. Check Input/Output Shapes
    session = ort.InferenceSession(MODEL_PATH)
    
    print("\n--- Model Inputs ---")
    for i in session.get_inputs():
        print(f"Name: {i.name}, Shape: {i.shape}, Type: {i.type}")

    print("\n--- Model Outputs ---")
    for i in session.get_outputs():
        print(f"Name: {i.name}, Shape: {i.shape}, Type: {i.type}")

    # 3. Run a Dummy Inference (The ultimate test)
    # Get input shape from model (assuming standard NCHW)
    input_shape = session.get_inputs()[0].shape
    # If dimensions are dynamic (None), fill them with standard values
    input_shape = [1 if d is None or isinstance(d, str) else d for d in input_shape]
    
    # Create dummy data (random noise)
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    # Run the model
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: dummy_input})
    
    print(f"\n✅ Inference successful! Model produced {len(outputs)} output(s).")

except Exception as e:
    print(f"\n❌ Error during verification: {e}")