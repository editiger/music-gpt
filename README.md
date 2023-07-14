<a href="https://colab.research.google.com/github/editigerun/guitarGPT/blob/main/guitarGPT.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# guitarGPT
=====


Making evaluating and fine-tuning LLaMA models with low-rank adaptation (LoRA) easy.


guitarGPT
=====

guitarGPT es un modelo entrenado para generar música simbólica para guitarra.

Características
--------

- El modelo está construído como un un ajuste fino del modelo de código abierto GPT-J.
- Se ha propuesto una tokenización simple y específica para guitarra basada en la estructura de una tablatura.
- El conjunto de datos son archivos musicXML que son procesados y deconstruídos en tokens tipo tablatura.
- Los resultados del modelo se transcriben a formato musicXML que puede ser leído por cualquier editor de música como TuxGuitar por ejemplo.
- A diferencia de otros modelos, se pueden tokenizar musica de cualquier duración de compás (2/4, 3/4, 4/4, 5/8, 9/8, 9/4, etc){

¿Por qué guitarGPT?
---------

En la literatura se han propuesto diversos modelos para la generación de música simbólica, todos ellos partiendo de archivos MIDI y usando herramientas como MIDITOK que tiene limitaciones como



Installation
------------

To install MusPy, please run `pip install muspy`. To build MusPy from source, please download the [source](https://github.com/salu133445/muspy/releases) and run `python setup.py install`.


Documentation
-------------

Documentation is available [here](https://salu133445.github.io/muspy) and as docstrings with the code.


Citing
------

Please cite the following paper if you use MusPy in a published work:

Hao-Wen Dong, Ke Chen, Julian McAuley, and Taylor Berg-Kirkpatrick, "MusPy: A Toolkit for Symbolic Music Generation," in _Proceedings of the 21st International Society for Music Information Retrieval Conference (ISMIR)_, 2020.

[[homepage](https://salu133445.github.io/muspy/)]
[[video](https://youtu.be/atdHMEuAYno)]
[[paper](https://salu133445.github.io/muspy/pdf/muspy-ismir2020-paper.pdf)]
[[slides](https://salu133445.github.io/muspy/pdf/muspy-ismir2020-slides.pdf)]
[[poster](https://salu133445.github.io/muspy/pdf/muspy-ismir2020-poster.pdf)]
[[arXiv](https://arxiv.org/abs/2008.01951)]
[[code](https://github.com/salu133445/muspy)]
[[documentation](https://salu133445.github.io/muspy/)]


Disclaimer
----------

This is a utility library that downloads and prepares public datasets. We do not host or distribute these datasets, vouch for their quality or fairness, or claim that you have license to use the dataset. It is your responsibility to determine whether you have permission to use the dataset under the dataset's license.

If you're a dataset owner and wish to update any part of it (description, citation, etc.), or do not want your dataset to be included in this library, please get in touch through a GitHub issue. Thanks for your contribution to the community!
