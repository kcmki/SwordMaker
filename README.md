# Minecraft Sword Generator

## Overview

This project aims to create a generative model capable of generating 16x swords for Minecraft. The diffusion model used was trained to produce unique and visually appealing sword designs.

The dataset was personally curated by collecting numerous diamond swords from various resource packs and resizing them to fit the desired dimensions.

I developed this project immediately after completing the Deeplearning.ai course on diffusion models. Big portion of the code is inspired by the examples used in their course.

## Code

The `Zip.py` script was utilized to extract needed swords textures from ressource packs zips .

The `imageTransformation.py` script was utilized for resizing the data.

All the needed classes are situated in diffusion_utilities and ContextUnet.

Made a sampling class in Sampling.py that will simplify sampling new pics and simplify the code in the UI.

To run the code, execute the command `python -m streamlit run Interface.py` to start the UI the will run using the weights stored in weights folder.

To train again the model if new data is added you can do it in the notebook `Training.ipynb` the new generated model will be stored in the weights folder.

For any further information or inquiries, feel free to contact me on Discord or Twitter.
