# import os
# os.chdir('normal')
# i=1
# for file in os.listdir():
#     src=file
#     dst="normal"+"_"+str(i)+".jpg"
#     os.rename(src,dst)
#     i+=1

import os
from PIL import Image

# 🔹 INPUT DATASET PATH
input_path = r"D:\aaa\Retinal Diesease\dataset"

# 🔹 OUTPUT PATH (processed dataset)
output_path = r"D:\aaa\Retinal Diesease\processed_dataset"

# Create output folder if not exists
os.makedirs(output_path, exist_ok=True)

# 🔹 Target size for ResNet
IMG_SIZE = (224, 224)

# Loop through each class folder
for class_name in os.listdir(input_path):
    class_input_path = os.path.join(input_path, class_name)
    
    # skip files (only folders)
    if not os.path.isdir(class_input_path):
        continue
    
    # create same class folder in output
    class_output_path = os.path.join(output_path, class_name)
    os.makedirs(class_output_path, exist_ok=True)
    
    count = 1
    
    for file in os.listdir(class_input_path):
        file_path = os.path.join(class_input_path, file)
        
        try:
            # open image
            img = Image.open(file_path).convert("RGB")
            
            # resize
            img = img.resize(IMG_SIZE)
            
            # new filename
            new_name = f"{class_name}_{count}.jpg"
            save_path = os.path.join(class_output_path, new_name)
            
            # save image
            img.save(save_path)
            
            count += 1
            
        except Exception as e:
            print(f"Skipped: {file} | Error: {e}")

print("✅ All images resized & renamed successfully!")