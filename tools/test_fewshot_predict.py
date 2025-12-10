import requests

URL = "http://127.0.0.1:5000/fewshot/predict"

img_path = r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\fakeface_generator\data\fake\run01\seed0034.png"  # real or fake

with open(img_path, "rb") as f:
    files = {"image": f}
    resp = requests.post(URL, files=files)

print(resp.status_code)
print(resp.json())
