# SpeechRecognition

End-to-end automatic speech recognition in Java.

Currently spectral analysis is based on the Lincoln Frontend which has proven to have desirable properties. Classification of speech is done using Hidden Markov Models.

Includes sample speech data for standard ARPAbet phonemes and digit words.

The project is still largely under development as there are numerical computation errors that have yet to be resolved during acoustic processing.

Future development will likely include:

- Continuous word recognition
- Integration of language and pronunciation modelling to improve accuracy
- Extending single Gaussian output probabilities to Gaussian Mixture Models
- More support for various speech-based applications