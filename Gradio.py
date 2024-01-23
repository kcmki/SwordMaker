import base64
import io
import cv2
import streamlit as st
from matplotlib import pyplot as plt
from ContextUnet import ContextUnet
import torch
from Sampling import Sampler
import numpy as np
from diffusion_utilities import norm_all, plot_sample, unorm
from IPython.display import HTML
import streamlit.components.v1 as components


# load in model weights and set to eval mode
# construct model
# network hyperparameters
save_dir = './weights/'

Sampler = Sampler()
#samples, interm = Sampler.sample_ddpm(1)
def DrawSamples(numberOfSamples):
    st.write("Started Sampling")
    samples, interm = Sampler.sample_ddpm(numberOfSamples)
    st.write("Finished Sampling")
       # unity norm to put in range [0,1] for np.imshow
    cols = st.columns(4)
    for i,sample in enumerate(samples) :
        # Create a Matplotlib figure
        fig, ax = plt.subplots(figsize=(1, 1))
        sample = sample.permute(1, 2, 0).cpu().numpy()
        sample = unorm(sample)
        # Plot the image on the Matplotlib axis
        ax.imshow(sample)

        # Turn off axis labels
        ax.axis('off')

        # Display the Matplotlib figure using Streamlit
        cols[i%4].pyplot(fig)
    st.write("Finished Displaying Samples")
    st.write("Started Animating Samples")
    animation_ddpm = plot_sample(interm,len(samples),len(samples)//4 + 1,save_dir, "ani_run", None, save=False)

    components.html(animation_ddpm.to_jshtml(),height=1000)
    st.write("Finished Animating Samples")
def main():
    # Display the image using Streamlit
    st.title("Sword Maker")
    st.sidebar.title("Controls")

    numberOfSamples = st.sidebar.number_input("Number of Samples", min_value=1, max_value=10, value=1, step=1, key="num_samples")

    if st.sidebar.button("Generate"):
        DrawSamples(numberOfSamples)

if __name__ == "__main__":
    main()