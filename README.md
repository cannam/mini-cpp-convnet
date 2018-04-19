
### mini-cpp-convnet

A small program to run (but not train) a convolutional network for
image classification, to accompany the blog post [What does a
convolutional neural net actually do when you run
it?](https://thebreakfastpost.com/2018/04/18/what-does-a-convolutional-neural-net-actually-do-when-you-run-it/)

The interesting code is in `flower.cpp`.

```
$ make
$ ./obtain-data.sh
$ ./flower ./data/test/sunflowers/9783416751_b2a03920f7_n.png 
Classification took 0.0849622 sec
sunflowers: 99.9181%
dandelion: 0.0560829%
tulips: 0.0258057%
daisy: 1.06907e-05%
roses: 2.91274e-06%
```

[![Build Status](https://travis-ci.org/cannam/mini-cpp-convnet.svg?branch=master)](https://travis-ci.org/cannam/mini-cpp-convnet)

Code by Chris Cannam, copyright Queen Mary University of
London. Published under MIT/X11 licence, see the file COPYING for
details.

