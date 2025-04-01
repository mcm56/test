
from image_classifier import ImageClassifier
from torchvision import models

# 自定义设置
classifier = ImageClassifier()

# 预测图像文件
results = classifier.predict("shi2.jpg")
for label, confidence in results:
    print(f"{label}: {confidence:.2%}")

