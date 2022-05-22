# Lua-QmDSP
QmDSP

# It's kind of a pain to build it
* Just get rid of the cblas/clapack
* It is missing a function anyhow
* The library doesn't call that but if it did it would crash
* Use Mkl instead

# So it has some cool stuff in it
* KaiserWindow
* Pitch
* SincWindow
* Windows
* Chromagrams
* KeyDetection
* MFCC
* onset detection
* phase vocoder
* rate conversion
* rhythym 
* segmentation
* hidden markov models
* signal stuff
* tempo tracker
* tonal stuff
* transforms like DFT and DCT
* Wavelets
* Correlation

# It uses some nested structures
* They can't be wrapped
* So I just decided to modfy the source a bit
* It's not a very huge library so I just took them out and renamed

# It's good for processing audio files
* it's not really going to be great for real-time
* It doesn't have any kind vectorization
* It might work though for small things

# It's a C++ wrapper
* There isn't any kind of documentation for qm-dsp that I found
* THe only documents are the header files really
* If you're not familiar with C++ this is going to be frustrating to use
* As it's just using Lua to script over the C++ 
