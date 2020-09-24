# OpenCV4.4.0在Windows x64下进行安装与编译


## 1、首先在Windows x64系统中安装MinGW-W64以支持C/C++编译需要
  下载地址: https://sourceforge.net/projects/mingw-w64/files/mingw-w64/mingw-w64-release/

  在这我们选择下载 MinGW-W64-install.exe, 这样在windiws下安装要方便些。

  安装的时，在弹出提示框 Architecture 选项，选择:x86_64

  然后将安装项目中mingw64/bin加入系统环境变量中，也是在环境设置的“PATH”中增加

  例如：
  ```
  D:\mingw-w64\x86_64-8.1.0-posix-seh-rt_v6-rev0\mingw64\bin
  ```

## 2、安装CMake

下载地址：https://cmake.org/download/

然后将安装CMake目录中的bin目录加入系统环境变量中

## 3、下载与编译OpenCV

这里已经用脚本实现整个过程, 复制以下脚本到文本文件中,将文件名称改为:opencv_build.bat,然后运行.

```sh
echo off

if not exist "C:\opencv" mkdir "C:\opencv"
if not exist "C:\opencv\build" mkdir "C:\opencv\build"

echo Downloading OpenCV sources
echo.
echo For monitoring the download progress please check the C:\opencv directory.
echo.

REM This is why there is no progress bar:
REM https://github.com/PowerShell/PowerShell/issues/2138

echo Downloading: opencv-4.4.0.zip [91MB]
powershell -command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri https://github.com/opencv/opencv/archive/4.4.0.zip -OutFile c:\opencv\opencv-4.4.0.zip"
echo Extracting...
powershell -command "$ProgressPreference = 'SilentlyContinue'; Expand-Archive -Path c:\opencv\opencv-4.4.0.zip -DestinationPath c:\opencv"
del c:\opencv\opencv-4.4.0.zip /q
echo.

echo Downloading: opencv_contrib-4.4.0.zip [58MB]
powershell -command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri https://github.com/opencv/opencv_contrib/archive/4.4.0.zip -OutFile c:\opencv\opencv_contrib-4.4.0.zip"
echo Extracting...
powershell -command "$ProgressPreference = 'SilentlyContinue'; Expand-Archive -Path c:\opencv\opencv_contrib-4.4.0.zip -DestinationPath c:\opencv"
del c:\opencv\opencv_contrib-4.4.0.zip /q
echo.

echo Done with downloading and extracting sources.
echo.

echo on

cd /D C:\opencv\build
set PATH=%PATH%;C:\Program Files (x86)\CMake\bin;C:\mingw-w64\x86_64-6.3.0-posix-seh-rt_v5-rev1\mingw64\bin
cmake C:\opencv\opencv-4.4.0 -G "MinGW Makefiles" -BC:\opencv\build -DENABLE_CXX11=ON -DOPENCV_EXTRA_MODULES_PATH=C:\opencv\opencv_contrib-4.4.0\modules -DBUILD_SHARED_LIBS=ON -DWITH_IPP=OFF -DWITH_MSMF=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_opencv_java=OFF -DBUILD_opencv_python=OFF -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF -DBUILD_DOCS=OFF -DENABLE_PRECOMPILED_HEADERS=OFF -DBUILD_opencv_saliency=OFF -DCPU_DISPATCH= -DOPENCV_GENERATE_PKGCONFIG=ON -DWITH_OPENCL_D3D11_NV=OFF -DOPENCV_ALLOCATOR_STATS_COUNTER_TYPE=int64_t -Wno-dev
mingw32-make -j%NUMBER_OF_PROCESSORS%
mingw32-make install
rmdir c:\opencv\opencv-4.4.0 /s /q
rmdir c:\opencv\opencv_contrib-4.4.0 /s /q
```


脚本成功运行，表示安装成功

然后将:C:\opencv\build\install\x64\mingw\bin 加入环境变量中。结束 