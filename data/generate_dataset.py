import cv2
import pandas as pd
import numpy as np

image = []
anno = []
for i in range(272, 32606):
    if i % 100 == 0:
        print(i)

    if i < 6332 or 13312 < i < 14868 or 19093 < i < 20024 or 25649 < i < 28997:
        img = cv2.imread('./train_images/%d.jpg' % i)
        image.append(img.transpose(2, 0, 1))
        anno.append(0)
    elif 6367 < i < 11984 or 14917 < i < 18792 or 20063 < i < 25416 or i > 30179:
        img = cv2.imread('./train_images/%d.jpg' % i)
        image.append(img.transpose(2, 0, 1))
        anno.append(1)

train_rate = 0.8
df = pd.DataFrame({'image': image, 'label': anno})
total = df.shape[0]
print(total)
train_num = int(total * train_rate)

df = df.sample(frac=1).reset_index(drop=True)
df_train = df.loc[0:train_num]
df_val = df.loc[train_num + 1:]

np.savez('train', images=df_train['image'].values, labels=df_train['label'].values)
np.savez('val', images=df_val['image'].values, labels=df_val['label'].values)
