# %%
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from skl2onnx import to_onnx
from onnxruntime import InferenceSession
from google.protobuf.json_format import MessageToJson

# %%
X, y = load_diabetes(return_X_y=True)
model = LinearRegression().fit(X, y)

onx = to_onnx(
    model=model,
    X=X.astype(np.float32),
    target_opset=12
)

# %%
# compare prediction difference
sess = InferenceSession(onx.SerializeToString())
yp1 = sess.run(output_names=None, input_feed={"X": X.astype(np.float32)})[0].ravel()
yp2 = model.predict(X.astype(np.float32))

print(f"std(difference)={np.std(yp1 - yp2):.1e}")

# %%
with open("sklearn_linear.json", "w") as f:
    f.write(MessageToJson(onx))

# %%
