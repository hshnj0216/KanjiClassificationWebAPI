# KanjiClassificationWebAPI

## Overview
This is an ASP.NET Core Web API that uses an ONNX model to classify submitted images of Kanji handwriting or hand-drawing. The model was trained with PyTorch on 1,235 classes or Kanji characters from KanjiAlive and 82,250 white stroke and black background images. The dataset was split using train-test-split technique, the model scored ~94% accuracy during the tests. 

## Input
The API accepts a single image data in binary format, for best results the image sent must be 224x224px with a white stroke and black background.

## Output 
The for the image classification, the output is a unicode string of the kanji in the "U+XXXX" format. For the class/character inference, the output is an array of unicode strings in the same format as ones in the image classification.

## Functionalities

### Image Classification
The web API takes a kanji character drawing with white stroke and black background and returns a unicode representation of the kanji character or class in the "U+XXXX" format.

### Class/Character inferrence
The web API takes a kanji character drawing with white stroke and black background and returns a list of 10 classes or kanji characters in unicode that the model inferred from the provided image in the "U+XXXX" format.
