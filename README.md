# Recreation and Improvement of a Spiking Neural Network Composer Classiﬁcation Temporal Coding

This repo aims to recreate the results presented in [1] and further improve on the authors methods using Spiking Neural Networks to perform classical composer music classification. This was done for the Brain Inspired Computing class (CMPE-765).

Project by:
* Enzo Casamassima
* Eri Montano
* Amado Pena

## Structure:

- `IEEE_BIC_paper.pdf` is the paper we wrote detailing our research work.
- `src/main.py` does the classification and generates the spiking plots.
- `src/dataprep.py` utility file that extracts the samples from the midi files.
- `src/Classical Piano Midis.zip` is the dataset. For the experiments, only Beethoven and Bach songs were used.

## Requirements:
- `pip install pretty_midi`
- [BindsNET (used PyTorch)](https://github.com/BindsNET/bindsnet)
- `pip install librosa` (optional)

References:
1. Saboo, K., & Rajendran, B. (2015, July). Composer classification based on temporal coding in adaptive spiking neural networks. In 2015 International Joint Conference on Neural Networks (IJCNN) (pp. 1-8). IEEE.
2. https://github.com/craffel/pretty-midi
3. https://notebook.community/craffel/pretty-midi/Tutorial