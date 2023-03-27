# Ubuntu 22 无法通过Publish over SSH 连接

问题： 当你需要通过Publish over SSH将jinkens编出的版本发布的线上服务器时，配置好所有配置后并确保密码（key）以及服务器相关信息都正确时，
jinkens怎么也连接不上线上服务器，怎么配置都会报:
```sh
 Failed to connect or change directory
 [Auth failed]
```

那么很有可能以下原因造成。

# 原因
由于各种安全漏洞，RSA SHA-1 哈希算法正迅速在操作系统和 SSH 客户端中被弃用，其中许多技术现在完全拒绝使用该算法。

这似乎发生ssh在 Ubuntu 22.04 的客户端上。RSA 公私密钥对不再被认为是安全的。

# 解决方案

使用更现代、更安全的密钥类型，例如ed25519. 使用以下命令在 Ubuntu 22.04 计算机中生成新的密钥对：

```sh
ssh-keygen -t ed25519 -C "colin@colin-desktop"
```

注意：后面的字符串-C是注释，通常将您的电子邮件地址放在这里。由于您可能只在家中使用此密钥，因此输入电子邮件地址可能没有意义。我会放一些像“colin@colin-desktop”这样的东西，这样你就知道密钥属于你家里的哪个用户和计算机。

或者，您可以使用 SHA2 哈希创建一个新的 RSA 密钥，如下所示：


```sh
ssh-keygen -t rsa-sha2-512 -b 4096
```

无论哪种方式，您都需要ssh-copy-id再次使用该命令：

```sh
ssh-copy-id root@xxx.xxx.xxx.xxx
```
此命令将识别有一个新的公钥并将新密钥复制到文件中/root/.ssh/authorized_keys。

它现在应该工作了。


# 远程服务器解决方法

如果您无法在本地计算机上更改任何内容，或者不想使用新密钥，并且想在本地计算机上重新启用 RSA，请编辑远程计算机上的文件并添加以下行/etc/ssh/sshd_config：

```sh
HostKeyAlgorithms +ssh-rsa
PubkeyAcceptedKeyTypes +ssh-rsa
```

这将允许使用您已有的不安全 RSA 密钥。

请记住通过以下方式重新启动 sshd 服务：


```sh
sudo systemctl restart sshd
```

否则，您必须重新启动计算机才能使更改生效。