import requests

infer_url = "https://aloshdenny--qwen2-5-vl-inference-infer.modal.run"

img_path = "/Users/aloshdenny/vscode/sandeep/uniform.jpg"
prompt = "Is there a person in the image, and are they wearing a uniform? Respond with YES or NO"  # <-- Change as needed

with open(img_path, "rb") as f:
    files = {"image": ("dress.jpg", f, "image/jpeg")}
    data = {"prompt": prompt}  # <-- Add prompt as form field
    r = requests.post(infer_url, files=files, data=data)

print(r.status_code)
print(r.json())