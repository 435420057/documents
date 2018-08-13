# 企业版APP通过URL地址进行下载
企业授权的APP想通过自己的URL地址进行下载，那么需要准备一个plist描述文件和一个ipa包。

### 1、 生成plist描述文件

plist描述文件类似于XML文件那样，如：
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>items</key>
    <array>
        <dict>
            <key>assets</key>
            <array>
                <dict>
                    <key>kind</key>
                    <string>software-package</string>
                    <key>url</key>
                    <string>https://url/app.ipa</string>
                </dict>
            </array>
            <key>metadata</key>
            <dict>
                <key>bundle-identifier</key>
                <string>com.xx.xx(应用程序的标准应用程序标识符字符串，在应用程序的.plist文件中指定)</string>
                <key>bundle-version</key>
                <string>1.0(应用版本号)</string>
                <key>kind</key>
                <string>software</string>
                <key>title</key>
                <string>xxx</string>
                <key>display-image</key>
                <!--a fully-qualified URL pointing to a 57×57-pixel (72x72 for iPad) PNG icon used during download and installation -->
                <string></string>
                <key>full-size-image</key>
                <!-- a fully-qualified URL pointing to a 512×512-pixel PNG image that represents the iTunes app -->
                <string></string>
            </dict>
        </dict>
    </array>
</dict>
</plist>
```

复制以上的格式的信息存在.plist文件，修改对应信息后保存。

### 2、 做准备一个WEB服务器

可以通过Nginx、IIS、Apache等创建自己的WEB服务器,绑定好相关域名后，将已经生成好的plist文件和ipa（关于ipa文件如何生成，这里就不多说了）上传到域名可以访问到的目录

### 3、 将下载信息绑定的下载按钮中

这步很简单只需这们：
```HTML
<a href="itms-services://?action=download-manifest&url=http://你的域名/xxx.plist">
  下载
</a>
```