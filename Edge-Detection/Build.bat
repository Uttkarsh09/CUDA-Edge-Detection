cls

@echo off

cd Bin/

nvcc.exe -c "../Source/CUDA/SobelCUDA.cu" -I "C:\opencv\build\include"
nvcc.exe -c "../Source/CUDA/CannyCUDA.cu"  -I "C:\opencv\build\include"

cl.exe /c /EHsc -I "C:\opencv\build\include" -I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\include" "../Source/OpenCV/SobelCV.cpp" "../Source/OpenCV/CannyCV.cpp" "../Source/Main.cpp"

link.exe Main.obj SobelCUDA.obj SobelCV.obj CannyCUDA.obj CannyCV.obj /LIBPATH:"C:\opencv\build\x64\vc16\lib" /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\lib\x64" cudart.lib opencv_world480.lib opencv_world480d.lib

@move Main.exe "../" > nul

cd ../
