cls

cl /c /EHsc -I "C:\opencv\build\include" SobelEdgeCV.cpp

nvcc -I "C:\opencv\build\include" -I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\include" -c SobelEdgeCUDA.cu

link SobelEdgeCV.obj /LIBPATH:"C:\opencv\build\x64\vc16\lib" opencv_world480.lib opencv_world480d.lib

link SobelEdgeCUDA.obj /LIBPATH:"C:\opencv\build\x64\vc16\lib" /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\lib\x64" cudart.lib opencv_world480.lib opencv_world480d.lib

SobelEdgeCV.exe

SobelEdgeCUDA.exe

