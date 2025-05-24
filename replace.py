import os

folder_path = '/home/ctai-manhdd10-d/Documents/ShortTextTopicModel/bash/NewMethod/top200'

for filename in os.listdir(folder_path):
    if 'top50' in filename:
        new_filename = filename.replace('top50', 'top200')
        
        old_file_path = os.path.join(folder_path, filename)
        new_file_path = os.path.join(folder_path, new_filename)
        
        os.rename(old_file_path, new_file_path)
        print(f"Đổi tên: {filename} -> {new_filename}")
