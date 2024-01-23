# -*- coding: utf-8 -*-

# Script to get all sword texture from inside texturepack zip files

from zipfile import ZipFile
import os
from PIL import Image
import re

def decode_escape_sequences(match):
    escape_sequence = match.group(0)
    try:
        decoded_character = chr(int(escape_sequence[2:], 16))
        return decoded_character
    except ValueError:
        return escape_sequence
    

def get_all_file_paths(directory):
  
    # initializing empty file paths list
    file_paths = []
  
    # crawling through directory and subdirectories
    for root, directories, files in os.walk(directory):
        for filename in files:
            # join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)
  
    # returning all file paths
    return file_paths        

directory_path = "D:\_dev\AI\diffusion\SwordMaker\Data\packs"


paths = get_all_file_paths(directory_path)
i=1
paths = [path for path in paths if path.endswith(".zip")]

#try:

for file_name in paths:
    print(file_name)
    try:
        with ZipFile(file_name,'r') as z:
            image = [file for file in z.namelist() if file.endswith("diamond_sword.png")]
            string = re.sub(r'\\x[0-9a-fA-F]+', decode_escape_sequences, image[0])
            image = z.extract(string)
            with Image.open(image) as img:
                img.save("D:\_dev\AI\diffusion\SwordMaker\Data\extracted\sw"+str(i)+".png")
            i=i+1
            z.close()
    except Exception as e:
        print(e)

"""         with ZipFile(file_name,'r') as z:
            try:
                image = [file for file in z.namelist() if file.endswith("diamond_sword.png")]
                string = re.sub(r'\\x[0-9a-fA-F]+', decode_escape_sequences, image[0])
                image = z.extract(string)
                with Image.open(image) as img:
                    img.save("D:\_dev\AI\diffusion\SwordMaker\Data\extracted\sw"+str(i)+".png")

            except Exception as e: 
                print(e)
                pass """
'''except Exception as e:
    print(e)
    pass'''
