# KanjiClassificationWebAPI

## Overview
This is an ASP.NET Core Web API that uses an ONNX model to classify submitted images of Kanji handwriting or hand-drawing. The model was trained with PyTorch on 1,235 classes or Kanji characters and 82,250 white stroke and black background images.

## Important note
For the moment, the ONNX model was not included in the repository as I have reached my GIT LFS limits.

## Functionalities

### Image Classification
The web API takes a kanji character drawing with white stroke and black background and returns a unicode representation of the kanji character or class in the "U+XXXX" format.

### Class/Character inferrence
The web API takes a kanji character drawing with white stroke and black background and returns a list of 10 classes or kanji characters in unicode that the model inferred from the provided image in the "U+XXXX" format.
