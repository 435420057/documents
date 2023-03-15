# Postgresql 14配置文件，配置项翻译及解释说明
```ini
#-----------------------------
# PostgreSQL configuration file
#-----------------------------
#
#This file consists of lines of the form:
#
#name = value
#
#(The "=" is optional.)  Whitespace may be used.  Comments are introduced with
#"#" anywhere on a line.  The complete list of parameter names and allowed
#values can be found in the PostgreSQL documentation.Shared Library Preloading
#
#The commented-out settings shown in this file represent the default values.
#Re-commenting a setting is NOT sufficient to revert it to the default value;
#you need to reload the server.
#
#This file is read on server startup and when the server receives a SIGHUP
#signal.  If you edit the file on a running system, you have to SIGHUP the
#server for the changes to take effect, run "pg_ctl reload", or execute
#"SELECT pg_reload_conf()".  Some parameters, which are marked below,
#require a server shutdown and restart to take effect.
#
#Any parameter can also be given as a command-line option to the server, e.g.,
#"postgres -c log_connections=on".  Some parameters can be changed at run time
#with the "SET" SQL command.
#
#Memory units:  B  = bytes            Time units:  us  = microseconds
#kB = kilobytes                     ms  = milliseconds
#MB = megabytes                     s   = seconds
#GB = gigabytes                     min = minutes
#TB = terabytes                     h   = hours
#d   = days


#------------------------------------------------------------------------------
## FILE LOCATIONS
#------------------------------------------------------------------------------
#除了已经提到的 postgresql.conf 文件之外，PostgreSQL 还使用了另外两个手动编辑的配置文件，它们控制客户端身份验证
#（它们的使用在第 21 章中讨论）。
#默认情况下，所有三个配置文件都存储在数据库集群的数据目录中。
#本节中描述的参数允许将配置文件放置在其他地方。 
#这样做可以简化管理。尤其是当配置文件分开保存时，通常更容易确保它们得到正确备份。）
#The default values of these variables are driven from the -D command-line
#option or PGDATA environment variable, represented here as ConfigDir.

#data_directory = 'ConfigDir'           # use data in another directory
                                        # (change requires restart)
					# 指定用于数据存储的目录。此参数只能在服务器启动时设置。
#hba_file = 'ConfigDir/pg_hba.conf'     # host-based authentication file
                                        # (change requires restart)
					# 指定基于主机的身份验证的配置文件（通常称为 pg_hba.conf）。
					# 此参数只能在服务器启动时设置。
#ident_file = 'ConfigDir/pg_ident.conf' # ident configuration file
                                        # (change requires restart)
					# 指定用户名映射的配置文件（通常称为 pg_ident.conf）。此参数只能在服务器启动时设置。

#If external_pid_file is not explicitly set, no extra PID file is written.
#external_pid_file = ''                 # write an extra PID file
                                        # (change requires restart)
					# 指定服务器应创建以供服务器管理程序使用的附加进程 ID (PID) 文件的名称。此参数只能在服务器启动时设置。



#在默认安装中，上述参数均未明确设置。相反，数据目录由 -D 命令行选项或 PGDATA 环境变量指定，配置文件都在数据目录中。
#如果您希望将配置文件保存在数据目录以外的其他位置，postgres -D 命令行选项或 PGDATA 环境变量必须指向包含配置文件的目录，
#并且 data_directory 参数必须在 postgresql.conf 中设置（或命令行）来显示数据目录的实际位置。
#请注意，data_directory 覆盖了数据目录位置的 -D 和 PGDATA，但没有覆盖配置文件的位置。

#如果您愿意，可以使用参数 config_file、hba_file 和/或 ident_file 分别指定配置文件的名称和位置。
#config_file (string)	指定主服务器配置文件（通常称为 postgresql.conf）。此参数只能在 postgres 命令行上设置。
#config_file 只能在 postgres 命令行中指定，但其他可以在主配置文件中设置。
#如果所有三个参数加上 data_directory 都被显式设置，则不需要指定 -D 或 PGDATA。

#设置任何这些参数时，将相对于启动 postgres 的目录解释相对路径。

#------------------------------------------------------------------------------
## CONNECTIONS AND AUTHENTICATION
#------------------------------------------------------------------------------

### - Connection Settings -

listen_addresses = '*'          # what IP address(es) to listen on;
                                        # comma-separated list of addresses;
                                        # defaults to 'localhost'; use '*' for all
                                        # (change requires restart)
										# 指定服务器用于侦听来自客户端应用程序的连接的 TCP/IP 地址。
										# 该值采用逗号分隔的主机名和/或数字 IP 地址列表的形式。
										# 特殊条目 * 对应于所有可用的 IP 接口。
										# 条目 0.0.0.0 允许监听所有 IPv4 地址， :: 允许监听所有 IPv6 地址。
										# 如果列表为空，则服务器根本不会侦听任何 IP 接口，在这种情况下，只能使用 Unix 域套接字连接到它。
										# 默认值为 localhost，它只允许建立本地 TCP/IP “环回”连接。
										# 虽然客户端身份验证（第 21 章）允许对谁可以访问服务器进行细粒度控制，但 listen_addresses 控制哪些接口接受连接尝试，
										# 这有助于防止在不安全的网络接口上重复的恶意连接请求。此参数只能在服务器启动时设置。
port = 1931                             # (change requires restart)
										# 服务器监听的 TCP 端口；默认为 5432。请注意，服务器侦听的所有 IP 地址都使用相同的端口号。此参数只能在服务器启动时设置。
max_connections = 100                   # (change requires restart)
										# 确定与数据库服务器的最大并发连接数。
										# 默认值通常为 100 个连接，但如果您的内核设置不支持它可能会更少（在 initdb 期间确定）。
										# 此参数只能在服务器启动时设置。
#superuser_reserved_connections = 3     # (change requires restart)  超级用户预留连接数
#unix_socket_directories = '/tmp'       # comma-separated list of directories 默认值通常是/ tmp，但可以在构建时更改。 
                                        # (change requires restart)
#unix_socket_group = ''                 # (change requires restart)  设置Unix域套接字的拥有组 默认情况下，这是空字符串，它使用服务器用户的默认组
#unix_socket_permissions = 0777         # begin with 0 to use octal notation  设置Unix域套接字的访问权限
                                        # (change requires restart)
#bonjour = off                          # advertise server via Bonjour  通过Bonjour启用服务器存在的广告。 默认是关闭的。
                                        # (change requires restart)
#bonjour_name = ''                      # defaults to the computer name  指定Bonjour服务名称。 如果此参数设置为空字符串”（这是默认值），则使用计算机名称。
                                        # (change requires restart)

### - TCP settings -
#see "man tcp" for details

#tcp_keepalives_idle = 0                # TCP_KEEPIDLE, in seconds; 指定TCP在发送Keepalive消息给客户端之后不活动的秒数。 值为0时使用系统默认值。 
                                        # 0 selects the system default 此参数仅在支持TCP_KEEPIDLE或等效套接字选项的系统上以及在Windows上受支持; 在其他系统上，
										#它必须是零。 
										# 在通过Unix域套接字连接的会话中，该参数将被忽略并始终读为零。
#tcp_keepalives_interval = 0            # TCP_KEEPINTVL, in seconds; 指定客户端未确认的TCP保持活动消息应重新传输的秒数。 值为0时使用系统默认值。 
										#此参数仅在支持TCP_KEEPINTVL或等效套接字选项的系统上以及在Windows上受支持; 在其他系统上，它必须是零。 
										#在通过Unix域套接字连接的会话中，该参数将被忽略并始终读为零。
                                        # 0 selects the system default
#tcp_keepalives_count = 0               # TCP_KEEPCNT;   指定在服务器连接到客户端之前可能丢失的TCP保持活动的数量已被终止。 
										#值为0时使用系统默认值。 此参数仅在支持TCP_KEEPCNT或等效套接字选项的系统上受支持; 在其他系统上，它必须是零。 
										#在通过Unix域套接字连接的会话中，该参数将被忽略并始终读为零。
                                        # 0 selects the system default
#tcp_user_timeout = 0                   # TCP_USER_TIMEOUT, in milliseconds;
                                        # 0 selects the system default
										# 在强制关闭 TCP 连接之前，指定传输数据可能保持未确认的时间量
										# 如果指定此值没有单位，则以毫秒为单位。值 0（默认值）选择操作系统的默认值。
										# 此参数仅在支持 TCP_USER_TIMEOUT 的系统上受支持；在其他系统上，它必须为零。
										# 在通过 Unix 域套接字连接的会话中，此参数被忽略并始终读取为零。

#client_connection_check_interval = 0   # time between checks for client
                                        # disconnection while running queries;
                                        # 0 for never

#### - Authentication -

#authentication_timeout = 1min          # 1s-600s  这个参数只能在postgresql.conf文件中被设置，它指定一个时间长度，在这个时间长度内，必须完成客户端认证操作，
										#否则客户端连接请求将被拒绝。它可以阻止某些客户端进行认证时长时间占用数据库连接。单位是秒，默认值是60。
#password_encryption = scram-sha-256    # scram-sha-256 or md5
#db_user_namespace = off				#此参数启用每个数据库用户名。它默认关闭。该参数只能在postgresql.conf文件或服务器命令行中设置。
										#如果这样，您应该将用户创建为username @ dbname。当连接客户端传递用户名时，@和数据库名称将附加到用户名中，
										#并且该服务器会查找特定于数据库的用户名。请注意，当您在SQL环境中创建名称包含@的用户时，您需要引用用户名。
										#启用此参数后，您仍然可以创建普通的全局用户。在客户端指定用户名时简单追加@，例如乔@。在用户名被服务器查找之前，@将被剥离。
										#db_user_namespace导致客户端和服务器的用户名表示方式不同。身份验证检查始终使用服务器的用户名完成，
										#因此必须为服务器的用户名配置身份验证方法，而不是客户端。因为md5在客户端和服务器上均使用用户名作为salt，
										#所以md5不能与db_user_namespace一起使用。


#### GSSAPI using Kerberos
#krb_server_keyfile = 'FILE:${sysconfdir}/krb5.keytab' 
										#设置Kerberos服务器密钥文件的位置。 该参数只能在postgresql.conf文件或服务器命令行中设置。
#krb_caseins_users = off				#设置是否应该区分大小写地处理GSSAPI用户名。 默认是关闭的（区分大小写）。 
										#该参数只能在postgresql.conf文件或服务器命令行中设置。

### - SSL -

#ssl = off  							#这个参数只有在启动数据库时，才能被设置。决定数据库是否接受SSL连接。默认值是off。
#ssl_ca_file = ''						#指定包含SSL服务器证书颁发机构（CA）的文件的名称。 缺省值为空，表示未加载CA文件，并且不执行客户端证书验证。 
										#（在以前的PostgreSQL版本中，这个文件的名称被硬编码为root.crt。）相对路径是相对于数据目录的。 
										#该参数只能在服务器启动时设置。
#ssl_cert_file = 'server.crt'			#指定包含SSL服务器证书的文件的名称。 缺省值是server.crt。 相对路径与数据目录有关。 该参数只能在服务器启动时设置。
#ssl_crl_file = ''						#指定包含SSL服务器证书吊销列表（CRL）的文件的名称。 缺省值为空，表示没有加载CRL文件。 
										#（在以前的PostgreSQL版本中，该文件的名称被硬编码为root.crl。）相对路径与数据目录相关。 该参数只能在服务器启动时设置。
#ssl_crl_dir = ''
#ssl_key_file = 'server.key'			#指定包含SSL服务器私钥的文件的名称。 缺省值是server.key。 相对路径与数据目录有关。 该参数只能在服务器启动时设置。
#ssl_ciphers = 'HIGH:MEDIUM:+3DES:!aNULL' # allowed SSL ciphers  #指定可以使用的SSL加密算法。
										#查看操作系统关于openssl的用户手册可以得到完整的加密算法列表（执行命令openssl ciphers –v也可以得到）。
#ssl_prefer_server_ciphers = on			# 指定是否使用服务器的 SSL 密码首选项，而不是客户端的。此参数只能在 postgresql.conf 文件或服务器命令行中设置。默认为开启。
										# 较旧的 PostgreSQL 版本没有此设置，并且始终使用客户端的首选项。此设置主要是为了向后兼容这些版本。
										# 使用服务器的首选项通常会更好，因为服务器配置得当的可能性更大。
#ssl_ecdh_curve = 'prime256v1'
#ssl_min_protocol_version = 'TLSv1.2'
#ssl_max_protocol_version = ''
#ssl_dh_params_file = ''
#ssl_passphrase_command = ''
#ssl_passphrase_command_supports_reload = off


#------------------------------------------------------------------------------
## RESOURCE USAGE (except WAL)
#------------------------------------------------------------------------------

### - Memory -

shared_buffers = 128MB                  # min 128kB		共享缓存区大小 不应超过总内存1/4  一般为1/8
                                        # (change requires restart)
#huge_pages = try                       # on, off, or try 当为try时，未开启大页或分配大页内存太小，数据库仍能启动，但不使用大页；当on时不能分配足够内存时，启动失败
                                        # (change requires restart)
#huge_page_size = 0                     # zero for system default
                                        # (change requires restart)
		#当使用huge_pages 启用大页面时，控制大页面的大小。默认为零 (0)。
		#当设置为 0 时，将使用系统上默认的超大页面大小。此参数只能在服务器启动时设置。
#temp_buffers = 8MB                     # min 800kB  
		# 临时表的缓存大小，默认为8M 通常保持默认值就可以
#max_prepared_transactions = 0          # zero disables the feature
                                        # (change requires restart)
#Caution: it is not advisable to set max_prepared_transactions nonzero unless
#you actively intend to use prepared transactions.
										#设置可以同时处于“准备”状态的最大事务数（请参阅 PREPARE TRANSACTION）。
										#将此参数设置为零（这是默认值）将禁用准备事务功能。
										#此参数只能在服务器启动时设置。
										#如果您不打算使用准备好的事务，则应将此参数设置为零以防止意外创建准备好的事务。
										#如果您使用准备好的事务，您可能希望 max_prepared_transactions 至少与 max_connections 一样大，以便每个会话都可以有一个准备好的事务待处理。
#work_mem = 4MB                         # min 64kB 为每个进程单独分配的内存，主要用于排序、HASH等操作
		#在写入临时磁盘文件之前，设置查询操作（例如排序或哈希表）使用的基本最大内存量。如果此值指定为不带单位，则以千字节为单位。
		#默认值为四兆字节 (4MB)。
		#请注意，对于复杂的查询，可能会并行运行多个排序或散列操作；在开始将数据写入临时文件之前，通常允许每个操作使用此值指定的内存量。
		#此外，几个正在运行的会话可以同时执行此类操作。因此，使用的总内存可能是 work_mem 值的许多倍；在选择值时必须牢记这一事实。
		#排序操作用于 ORDER BY、DISTINCT 和合并连接。哈希表用于哈希连接、基于哈希的聚合、结果缓存节点和基于哈希的 IN 子查询处理。

基于哈希的操作通常比等效的基于排序的操作对内存可用性更敏感。可用于哈希表的内存是通过将 work_mem 乘以 hash_mem_multiplier 来计算的。这使得基于散列的操作可以使用超过通常 work_mem 基本数量的内存量。
#hash_mem_multiplier = 1.0              # 1-1000.0 multiplier on hash table work_mem 
										#用于计算基于哈希的操作可以使用的最大内存量。
										#最终限制是通过将 work_mem 乘以 hash_mem_multiplier 来确定的。
										#默认值为 1.0，这使得基于散列的操作服从与基于排序的操作相同的简单 work_mem 最大值。
#maintenance_work_mem = 64MB            # min 1MB
										#指定维护操作使用的最大内存量，例如 VACUUM、CREATE INDEX 和 ALTER TABLE ADD FOREIGN KEY。
										#如果此值指定为不带单位，则以千字节为单位。默认为 64 兆字节 (64MB)。
										#由于数据库会话一次只能执行其中一个操作，并且安装通常不会同时运行许多操作，因此将此值设置为远大于 work_mem 是安全的。
										#较大的设置可能会提高清理和还原数据库转储的性能。
										#请注意，当 autovacuum 运行时，最多可能会分配 autovacuum_max_workers 次此内存，因此请注意不要将默认值设置得太高。
										#通过单独设置 autovacuum_work_mem 来控制这一点可能很有用。
										#请注意，对于死元组标识符的收集，VACUUM 最多只能使用 1GB 的内存。
#autovacuum_work_mem = -1               # min 1MB, or -1 to use maintenance_work_mem
										#指定每个 autovacuum 工作进程要使用的最大内存量。
										#如果此值指定为不带单位，则以千字节为单位。它默认为 -1，表示应该使用 maintenance_work_mem 的值。
										#当在其他上下文中运行时，该设置对 VACUUM 的行为没有影响。
										#此参数只能在 postgresql.conf 文件或服务器命令行中设置。
										#对于死元组标识符的收集，autovacuum 最多只能使用 1GB 内存，
										#因此将 autovacuum_work_mem 设置为高于该值的值不会影响 autovacuum 在扫描表时可以收集的死元组数量。
										
#logical_decoding_work_mem = 64MB       # min 64kB
										#指定在将某些已解码更改写入本地磁盘之前，逻辑解码要使用的最大内存量。
										#这限制了逻辑流复制连接使用的内存量。
										#默认为 64 兆字节 (64MB)。
										#由于每个复制连接仅使用此大小的单个缓冲区，并且安装通常不会同时具有许多此类连接（受 max_wal_senders 限制），
										#因此将此值设置为明显高于 work_mem 是安全的，从而减少写入的解码更改量到磁盘。
										
#max_stack_depth = 2MB                  # min 100kB
										#指定服务器执行堆栈的最大安全深度。
										#此参数的理想设置是内核强制执行的实际堆栈大小限制（由 ulimit -s 或本地等效项设置），减去大约 1 兆字节的安全余量。
										#需要安全边际，因为堆栈深度不会在服务器中的每个例程中检查，而只是在关键的潜在递归例程中检查。
										#如果此值指定为不带单位，则以千字节为单位。默认设置为 2 兆字节 (2MB)，相对较小，不太可能导致崩溃。
										#但是，它可能太小而无法执行复杂的功能。只有超级用户可以更改此设置。
										#将 max_stack_depth 设置为高于实际内核限制将意味着失控的递归函数可能会使单个后端进程崩溃。
										#在 PostgreSQL 可以确定内核限制的平台上，服务器不允许将此变量设置为不安全的值。
										#但是，并非所有平台都提供信息，因此建议在选择值时谨慎。
										
#shared_memory_type = mmap              # the default is the first option
                                        # supported by the operating system:
                                        #   mmap
                                        #   sysv
                                        #   windows
                                        # (change requires restart)
										#指定服务器应该用于保存 PostgreSQL 的共享缓冲区和其他共享数据的主共享内存区域的共享内存实现。
										#可能的值是 mmap（用于使用 mmap 分配的匿名共享内存）、
										#sysv（用于通过 shmget 分配的 System V 共享内存）和 windows（用于 Windows 共享内存）。
										#并非所有平台都支持所有值；第一个支持的选项是该平台的默认选项。
										#通常不鼓励使用 sysv 选项，这在任何平台上都不是默认选项，因为它通常需要非默认内核设置来允许大分配（参见第 19.4.1 节）。
										
dynamic_shared_memory_type = posix      # the default is the first option
                                        # supported by the operating system:
                                        #   posix
                                        #   sysv
                                        #   windows
                                        #   mmap
                                        # (change requires restart)
										# 指定服务器应该使用的动态共享内存实现。
										# 可能的值是 posix（用于使用 shm_open 分配的 POSIX 共享内存）、
										# sysv（用于通过 shmget 分配的 System V 共享内存）、
										# windows（用于 Windows 共享内存）和 
										# mmap（使用存储在数据中的内存映射文件模拟共享内存目录）。
										# 并非所有平台都支持所有值；第一个支持的选项是该平台的默认选项。
										# mmap 选项在任何平台上都不是默认选项，通常不鼓励使用，
										# 因为操作系统可能会反复将修改后的页面写回磁盘，从而增加系统 I/O 负载；
										# 但是，当 pg_dynshmem 目录存储在 RAM 磁盘上或其他共享内存设施不可用时，它可能对调试有用。
										
#min_dynamic_shared_memory = 0MB        # (change requires restart)
										# 指定应在服务器启动时分配以供并行查询使用的内存量。
										# 当此内存区域不足或被并发查询耗尽时，新的并行查询会尝试使用配置了 dynamic_shared_memory_type 的方法临时
										# 从操作系统分配额外的共享内存，由于内存管理开销，这可能会更慢。
										# 在启动时使用 min_dynamic_shared_memory 分配的内存受支持的操作系统上的 huge_pages 设置的影响，
										# 并且可能更有可能从自动管理的操作系统上的较大页面中受益。默认值为 0（无）。此参数只能在服务器启动时设置。

### - Disk -

#temp_file_limit = -1                   # limits per-process temp file space
                                        # in kilobytes, or -1 for no limit
										# 指定进程可用于临时文件（例如排序和散列临时文件）或保留游标的存储文件的最大磁盘空间量。
										# 试图超过此限制的交易将被取消。如果此值指定为不带单位，则以千字节为单位。 -1（默认）表示没有限制。
										# 只有超级用户可以更改此设置。
										# 此设置限制给定 PostgreSQL 进程使用的所有临时文件在任何时刻使用的总空间。
										# 应该注意的是，用于显式临时表的磁盘空间，而不是在查询执行中在幕后使用的临时文件，不计入此限制。

### - Kernel Resources - 内核资源使用

#max_files_per_process = 1000           # min 64
                                        # (change requires restart)
										# 设置每个服务器子进程允许同时打开的文件的最大数量。
										# 默认值为一千个文件。如果内核强制执行每个进程的安全限制，则无需担心此设置。
										# 但是在某些平台上（特别是大多数 BSD 系统），如果许多进程都试图打开那么多文件，
										# 内核将允许单个进程打开比系统实际支持的更多的文件。
										# 如果您发现自己看到“打开的文件太多”失败，请尝试减少此设置。此参数只能在服务器启动时设置。

### - Cost-Based Vacuum Delay -  基于代价的清理延迟
										# 在VACUUM和ANALYZE命令的执行过程中，系统维持着一个内部计数器来跟踪各种被执行的I/O操作的估算开销。
										# 当累计的代价达到一个限制（由vacuum_cost_limit指定），执行这些操作的进程将按照vacuum_cost_delay所指定的休眠一小段时间。
										# 然后它将重置计数器并继续执行。
										# 这个特性的出发点是允许管理员降低这些命令对并发的数据库活动产生的I/O影响。
										# 在很多情况下，VACUUM和ANALYZE等维护命令能否快速完成并不重要，
										# 而非常重要的是这些命令不会对系统执行其他数据库操作的能力产生显著的影响。
										# 基于代价的清理延迟提供了一种方式让管理员能够保证这一点。

										# 对于手动发出的VACUUM命令，该特性默认被禁用。要启用它，只要把vacuum_cost_delay变量设为一个非零值。
#vacuum_cost_delay = 0                  # 0-100 milliseconds (0 disables)
										# 当超出开销限制时进程将要休眠的时间量。如果指定值时没有单位，则以毫秒为单位。 
										# 其默认值为0，这将禁用基于代价的清理延迟特性。正值将启用基于代价的清理。
										# 在使用基于代价的清理时，vacuum_cost_delay的合适值通常很小，也许是小于1毫秒。 
										# 虽然vacuum_cost_delay可以被设置为毫秒级别的值，但是在较老的平台上可能无法准确地测量这种延迟。 
										# 在这样的平台上，增加 VACUUM的节流资源消耗在1ms以上，需要改变其他的清理开销参数。 
										# 尽管如此，你应该保持 vacuum_cost_delay 在平台能持续测量的情况下尽可能小；大延迟没有帮助。
#vacuum_cost_page_hit = 1               # 0-10000 credits
										# 清理一个在共享缓存中找到的缓冲区的估计代价。它表示锁住缓冲池、查找共享哈希表和扫描页内容的代价。默认值为1。
#vacuum_cost_page_miss = 2              # 0-10000 credits
										# 清理一个必须从磁盘上读取的缓冲区的代价。它表示锁住缓冲池、查找共享哈希表、从磁盘读取需要的块以及扫描其内容的代价。
										# 默认值为10。
#vacuum_cost_page_dirty = 20            # 0-10000 credits
										# 当清理修改一个之前干净的块时需要花费的估计代价。它表示再次把脏块刷出到磁盘所需要的额外I/O。默认值为20。
#vacuum_cost_limit = 200                # 1-10000 credits
										# 将导致vacuum过程休眠的累积成本。默认值为 200。
										
										# 有些操作会保持关键性的锁，这样可以尽快完成。基于代价的清理延迟在这类操作期间不会发生。
										# 因此有可能代价会累计至大大超过指定的限制。
										# 为了防止在这种情况下的无意义的长时间延迟，
										# 实际延迟的计算方式是 vacuum_cost_delay * accumulated_balance / vacuum_cost_limit，且最大值是 vacuum_cost_delay * 4。

### - Background Writer -
										# 有一个独立的服务器进程，叫做后台写入器，它的功能就是发出写“脏”（新的或修改过的）共享缓冲区的命令。
										# 它写出共享缓冲区，这样让处理用户查询的服务器进程很少或者永不等待写动作的发生。
										# 不过，后台写入器确实会增加 I/O 的总负荷，因为虽然在每个检查点间隔中一个重复弄脏的页面可能只会写出一次，
										# 但在同一个间隔中后台写入器可能会把它写出好几次。在这一小节讨论的参数可以被用于调节本地需求的行为。
#bgwriter_delay = 200ms                 # 10-10000ms between rounds
										# 指定后台写入器活动轮次之间的延迟。在每个轮次中，写入器都会为一定数量的脏缓冲区发出写操作（可以用下面的参数控制）。 
										# 然后它就休眠 bgwriter_delay的时长， 然后重复动作。
										# 当缓冲池中没有脏缓冲区时，不管 bgwriter_delay，它都会进入更长的休眠。
										# 如果指定值时没有单位，则以毫秒为单位。默认值是 200 毫秒（200ms）。 
										# 注意在许多系统上，休眠延迟的有效解析度是 10 毫秒；
										# 因此，为bgwriter_delay设置一个 不是 10 的倍数的值与把它设置为下一个更高的 10 的倍数是一样的效果。
										# 这个选项只能在服务器命令行上或者在postgresql.conf文件中设置。
#bgwriter_lru_maxpages = 100            # max buffers written/round, 0 disables
										# 在每个轮次中，不超过这么多个缓冲区将被后台写入器写出。
										# 把这个参数设置为零可禁用后台写出（注意被一个独立、专用辅助进程管理的检查点不受影响）。
										# 默认值是 100 个缓冲区。这个参数只能在postgresql.conf文件中或在服务器命令行上设置。
#bgwriter_lru_multiplier = 2.0          # 0-10.0 multiplier on buffers scanned/round
										# 每一轮次要写的脏缓冲区的数目基于最近几个轮次中服务器进程需要的新缓冲区的数目。 
										# 最近所需的平均值乘以bgwriter_lru_multiplier可以估算下一轮次中将会需要的缓冲区数目。
										# 脏缓冲区将被写出直到有很多干净可重用的缓冲区（然而，每一轮次中写出的缓冲区数不超过bgwriter_lru_maxpages）。 
										# 因此，设置为 1.0 表示一种“刚刚好的”策略，这种策略会写出正好符合预测值的数目的缓冲区。 
										# 更大的值可以为需求高峰提供某种缓冲，而更小的值则需要服务进程来处理一些写出操作。
										# 默认值是 2.0。这个参数只能在postgresql.conf文件中或在服务器命令行上设置。
#bgwriter_flush_after = 512kB           # measured in pages, 0 disables
										# 只要后台写入的数据超过这个数量，尝试强制 OS 把这些写发送到底层存储上。
										# 这样做将限制内核页缓存中脏数据的量，降低了在检查点末尾发出一个 fsync 时或者 OS 在后台大批量写回数据时卡住的可能性。
										# 那常常会导致大幅度压缩的事务延迟，但是也有一些情况（特别是负载超过shared_buffers但小于 OS 页面高速缓存）的性能会降低。
										# 这种设置可能会在某些平台上没有效果。 
										# 如果指定值时没有单位，则以块为单位，即为BLCKSZ 字节，通常为8kB.合法的范围在0（禁用受控写回）和2MB之间。
										# Linux 上的默认值是512kB，其他平台上是0（如果BLCKSZ不是8kB，则默认值和最大值会按比例缩放至这个值）。
										# 这个参数只能在postgresql.conf文件中或者服务器命令行上设置。

### - Asynchronous Behavior -

#backend_flush_after = 0                # measured in pages, 0 disables
										# 只要一个后端写入的数据量超过这个数量时，就会尝试强制 OS 把这些写发送到底层存储。
										# 这样做将会限制内核页高速缓存中的脏数据数量，降低在检查点末尾发出fsync时或者 OS 在后台大批写回数据时卡住的可能性。
										# 这常常会导致极大降低的事务延迟，但是也有一些情况中（特别是负载超过shared_buffers但低于 OS 的页面高速缓存时），
										# 性能可能会下降。这个设置可能在某些平台上没有效果。 
										# 如果指定值时没有单位，则以块为单位，即为 BLCKSZ 字节，通常为8kB。
										# 合法的范围位于0（禁用受控写回）和2MB之间。默认是0（即没有强制写回）。（如果BLCKSZ不是8kB，最大值会按比例缩放到它）。
#effective_io_concurrency = 1           # 1-1000; 0 disables prefetching
										# 设置PostgreSQL可以同时被执行的并发磁盘 I/O 操作的数量。
										# 调高这个值，可以增加任何单个PostgreSQL会话试图并行发起的 I/O 操作的数目。 
										# 允许的范围是 1 到 1000，或 0 表示禁用异步 I/O 请求。当前这个设置仅影响位图堆扫描。

										# 对于磁盘驱动器，这个设置的一个很好的出发点是组成一个被用于该数据库的 RAID 0 条带或 RAID 1 镜像的独立驱动器数量
										# （对 RAID 5 而言，校验驱动器不计入）。
										# 但是， 如果数据库经常忙于在并发会话中发出的多个查询，较低的值可能足以使磁盘阵列繁忙。
										# 比保持磁盘繁忙所需的值更高的值只会造成额外的 CPU 开销。
										# SSD 以及其他基于内存的存储常常能处理很多并发请求，因此它们的最佳值可能是数百。

										# 异步 I/O 依赖于一个有效的posix_fadvise函数（一些操作系统可能没有）。 
										# 如果不存在这个函数，将这个参数设置为除 0 之外的任何东西将导致错误。
										# 在一些操作系统上（如Solaris）虽然提供了这个函数，但它不会做任何事情。

										# 在支持的系统上默认值为 1，否则为 0。
										# 对于一个特定表空间中的表，可以通过设定该表空间的同名参数（见ALTER TABLESPACE）可以覆盖这个值。
#maintenance_io_concurrency = 10        # 1-1000; 0 disables prefetching
										# 与effective_io_concurrency相似，但用于支持许多客户端会话完成的维护工作。
										# 在支持的系统上的默认值为 10，否则为 0。 
										# 对于特定表空间中的表，这个值可以被覆盖，通过设置同名的表空间参数（参见 ALTER TABLESPACE）。
#max_worker_processes = 8               # (change requires restart)
										# 设置系统能够支持的后台进程的最大数量。这个参数只能在服务器启动时设置。默认值为 8。
										# 在运行一个后备服务器时，你必须把这个参数设置为等于或者高于主控服务器上的值。否则， 后备服务器上可能不会允许查询。
										# 在更改这个值时，考虑也对max_parallel_workers、max_parallel_maintenance_workers以及max_parallel_workers_per_gather进行调整。
#max_parallel_workers_per_gather = 2    # taken from max_parallel_workers
										# 设置单个Gather或者Gather Merge节点能够开始的工作者的最大数量。
										# 并行工作者会从max_worker_processes建立的进程池中取得，数量由max_parallel_workers限制。
										# 注意所要求的工作者数量在运行时可能实际无法被满足。
										# 如果这种事情发生，该计划将会以比预期更少的工作者运行，这可能会不太高效。默认值是2。
										# 把这个值设置为0将会禁用并行查询执行。
#max_parallel_maintenance_workers = 2   # taken from max_parallel_workers
										# 设置单一工具性命令能够启动的并行工作者的最大数目。 
										# 当前，支持使用并行工作者的工具性命令是CREATE INDEX，并且只有在构建B-树索引时才能并行，并且 VACUUM 没有 FULL选项。 
										# 并行工作者从由max_worker_processes创建的进程池中取出，数量由max_parallel_workers控制。 
										# 注意实际在运行时所请求数量的工作者可能不可用。如果发生这种情况，工具性操作将使用比预期数量少的工作者运行。
										# 默认值为2。将这个值设置为0可以禁用工具性命令对并行工作者的使用。

										# 注意并行工具性命令不应该消耗比同等数量非并行操作更多的内存。
										# 这种策略与并行查询不同，并行查询的资源限制通常是应用在每个工作者进程上。
										# 并行工具性命令把资源限制maintenance_work_mem当作对整个工具性命令的限制，而不管其中用到了多少个并行工作者进程。
										# 不过，并行工具性命令实际上可能仍会消耗更多的CPU资源和I/O带宽。
#max_parallel_workers = 8               # maximum number of max_worker_processes that
                                        # can be used in parallel operations
										# 设置系统为并行操作所支持的工作者的最大数量。
										# 默认值为8。
										# 在增加或者减小这个值时，也要考虑对max_parallel_maintenance_workers以及max_parallel_workers_per_gather进行调整。
										# 此外，要注意将这个值设置得大于max_worker_processes将不会产生效果，因为并行工作者进程都是从max_worker_processes所建立的工作者进程池中取出来的。
#parallel_leader_participation = on
										# Gather允许领导进程在和节点下执行查询计划，Gather Merge而不是等待工作进程。
										# 默认值为on. 将此值设置为off可降低由于领导者读取元组的速度不够快而导致工作人员阻塞的可能性，
										# 但需要领导进程等待工作进程启动，然后才能生成第一个元组。
										# 领导者可以帮助或阻碍性能的程度取决于计划类型、工作人员数量和查询持续时间。
#old_snapshot_threshold = -1            # 1min-60d; -1 disables; 0 is immediate
                                        # (change requires restart)
										# 设置在使用快照时，一个快照可以被使用而没有发生“snapshot too old” 错误的风险的最小时间。
										# 超过此阈值时间的死数据将允许被清除。 这可以有助于阻止长时间使用的快照造成的快照膨胀。
										# 为了阻止由于本来对该快照可见的数据被清理导致的不正确结果，
										# 当快照比这个阈值更旧并且该快照被用来读取一个该快照建立以来被修改过的页面时，将会产生一个错误。

										# 如果指定值时没有单位，则以分钟为单位。 
										# 值 -1 (默认值) 禁用此功能,实际上将快照的时限设置为无穷大。 这个参数只能在服务器启动时设置。

										# 对于生产工作有用的值的范围可能从几个小时到几天不等。只允许小的值（例如0或者1min）是因为它们有时可能对测试有用。 
										# 虽然允许高达60d的设置，但是请注意很多负载情况下，很短的时间帧里就可能发生极大的膨胀或者事务 ID 回卷。

										# 当这个特性被启用时，关系末尾的被清出的空间不能被释放给操作系统，因为那可能会移除用于检测“snapshot too old”情况所需的信息。
										# 所有分配给关系的空间还将与该关系关联在一起便于重用，除非它们被显式地释放（例如，用VACUUM FULL）。

										# 这个设置不会尝试保证在任何特殊情况下都会生成错误。
										# 事实上，如果（例如）可以从一个已经物化了一个结果集的游标中生成正确的结果，
										# 即便被引用表中的底层行已经被清理掉也不会生成错误。某些表不能被过早地安全清除，并且因此将不受这个设置的影响，例如系统目录。
										# 对于这些表，这个设置将不能降低膨胀，也不能降低在扫描时产生“snapshot too old”错误的可能性。

#------------------------------------------------------------------------------
## WRITE-AHEAD LOG
#------------------------------------------------------------------------------

### - Settings -

wal_level = replica                     # minimal, replica, or logical
                                        # (change requires restart)
										# wal_level决定多少信息写入到 WAL 中。
										# 默认值是replica，它会写入足够的数据以支持WAL归档和复制，包括在后备服务器上运行只读查询。
										# minimal会去掉除从崩溃或者立即关机中进行恢复所需的信息之外的所有记录。最后，logical会增加支持逻辑解码所需的信息。
										# 每个层次包括所有更低层次记录的信息。这个参数只能在服务器启动时设置。
										# 在logical层，与replica相同的信息会被记录，外加上 允许从 WAL 抽取逻辑修改集所需的信息。
										# 使用级别 logical将增加 WAL 容量，特别是如果为了REPLICA IDENTITY FULL配置了很多表并且执行了很多UPDATE和DELETE 语句时。

										# 在 9.6 之前的版本中，这个参数也允许值archive和hot_standby。现在仍然接受这些值，但是它们会被映射到replica。
#fsync = on                             # flush data to disk for crash safety
                                        # (turning this off can cause
                                        # unrecoverable data corruption)
#synchronous_commit = on                # synchronization level;
                                        # off, local, remote_write, remote_apply, or on
										# 指定数据库服务器返回“success”指示给客户端之前，必须要完成多少WAL处理。 合法的值为remote_apply, on(默认值), remote_write,local, 和 off。
										# 如果synchronous_standby_names为空，则唯一有意义的设置为on 和 off ； 
										# remote_apply，remote_write 和 local都提供与on相同的本地同步级别。 
										# 所有非off模式的本地行为都是等待WAL的本地刷新到磁盘。 
										# 在 off模式，无需等待，因此在向客户端报告成功和以后保证事务安全防止服务器崩溃之间可能会出现延迟。 
										# 当设置为off时，在向客户端报告成功和真正保证事务不会被服务器崩溃威胁之间会有延迟（最大的延迟是wal_writer_delay的三倍）。 
										# 不同于fsync，将这个参数设置为off不会产生数据库不一致性的风险：
										# 一个操作系统或数据库崩溃可能会造成一些最近据说已提交的事务丢失，但数据库状态是一致的，就像这些事务已经被干净地中止。 
										# 因此，当性能比完全确保事务的持久性更重要时，关闭synchronous_commit可以作为一个有效的代替手段。更多讨论见第 30.3 节。

										# 如果synchronous_standby_names为非空，synchronous_commit也控制是否事务提交将等待它们的 WAL 记录在后备服务器上被处理。

										# 当设置为 remote_apply 时，提交将等待，直到来自当前同步备用服务器的答复显示他们已收到事务的提交记录并应用了它，
										# 以便它变得对备用服务器上的查询可见，并写入备用服务器上的持久存储。 
										# 这将导致比以前的设置更大的提交延迟，因为它等待 WAL 重放(replay)。 
										# 当设置为on时，提交将等待，直到来自于当前同步的后备服务器的回复显示它们已经收到了事务的提交记录并将其刷入了磁盘。 
										# 这保证事务将不会被丢失，除非主服务器和所有同步后备都遭受到了数据库存储损坏的问题。 
										# 当这个参数被设置为remote_write时，提交将等待，直到来自当前的同步后备的回复指示它们已经收到了该事务的提交记录
										# 并且已经把该记录写到它们的文件系统，这种设置保证数据得以保存，在PostgreSQL的后备服务器实例崩溃时，
										# 但是不能保证后备服务器遭受操作系统级别崩溃时数据能被保持，因为数据不一定必须要在后备机上达到持久存储。 
										# 设置local会导致提交等待本地刷写到磁盘，而不是复制。在使用同步复制时这通常是不可取的，但是为了完整性提供了这个选项。

										# 这个参数可以随时被修改；任何一个事务的行为由其提交时生效的设置决定。
										# 因此，可以同步提交一些事务，同时异步提交其他事务。
										# 例如，当默认是相反时，实现一个单一多语句事务的异步提交，在事务中发出SET LOCAL synchronous_commit TO OFF。
#wal_sync_method = fsync                # the default is the first option
                                        # supported by the operating system:
                                        #   open_datasync  用open()选项O_DSYNC写 WAL 文件
                                        #   fdatasync (default on Linux and FreeBSD)  在每次提交时调用fdatasync()
                                        #   fsync    在每次提交时调用fsync()
                                        #   fsync_writethrough  在每次提交时调用fsync()，强制任何磁盘写高速缓存的直通写
                                        #   open_sync   用open()选项O_SYNC写 WAL 文件
										# 用来向强制 WAL 更新到磁盘的方法。如果fsync是关闭的，那么这个设置就不相关，因为 WAL 文件更新将根本不会被强制。
full_page_writes = on                   # recover from partial page writes
										# 当这个参数为打开时，PostgreSQL服务器在一个检查点之后的页面的第一次修改期间将每个页面的全部内容写到 WAL 中。
										# 这么做是因为在操作系统崩溃期间正在处理的一次页写入可能只有部分完成，从而导致在一个磁盘页面中混合有新旧数据。
										# 在崩溃后的恢复期间，通常存储在 WAL 中的行级改变数据不足以完全恢复这样一个页面。
										# 存储完整的页面映像可以保证页面被正确存储，但代价是增加了必须被写入 WAL 的数据量（因为 WAL 重放总是从一个检查点开始，
										# 所以在检查点后每个页面的第一次改变时这样做就够了。因此，一种减小全页面写开销的方法是增加检查点间隔参数值）。

										# 把这个参数关闭会加快正常操作，但是在系统失败后可能导致不可恢复的数据损坏，或者静默的数据损坏。
										# 其风险类似于关闭fsync， 但是风险较小。并且只有在可关闭fsync的情况下才应该关闭它。

										# 关闭这个选项并不影响用于时间点恢复（PITR）的 WAL 归档使用（见第 26.3 节）。

										# 这个参数只能在postgresql.conf文件中或在服务器命令行上设置。默认值是on。
										
#wal_log_hints = off                    # also do full page writes of non-critical updates
                                        # (change requires restart)
										# 当这个参数为on时，PostgreSQL服务器一个检查点之后页面被第一次修改期间把该磁盘页面的整个内容都写入 WAL，
										# 即使对所谓的提示位做非关键修改也会这样做。

										# 如果启用了数据校验和，提示位更新总是会被 WAL 记录并且这个设置会被忽略。
										# 你可以使用这个 设置测试如果你的数据库启用了数据校验和，会有多少额外的 WAL 记录发生。

										# 这个参数只能在服务器启动时设置。默认值是off。
										
#wal_compression = off                  # enable compression of full-page writes
										# 当这个参数为on时，如果full_page_writes 为打开或者处于基础备份期间，PostgreSQL服务器 会压缩写入到 WAL 中的完整页面镜像。
										# 压缩页面镜像将在 WAL 重放时 被解压。默认值为off。只有超级用户可以更改这个设置。

										# 打开这个参数可以减小 WAL 所占的空间且无需承受不可恢复的数据损坏风险， 但是代价是需要额外的 CPU 开销以便在 WAL 记录期间
										# 进行压缩以及在 WAL 重放时解压。
										
#wal_init_zero = on                     # zero-fill new WAL files
										# 如果设置为on（默认值），此选项会导致新的 WAL 文件被零填充。 
										# 在某些文件系统上，这可确保在我们需要写入 WAL 记录之前分配空间。 但是，Copy-On-Write（COW）文件系统
										# 可能不会从此技术中受益，因此可以选择跳过不必要的工作。 
										# 如果设置为off，则在创建文件时仅写入最终字节，以便其具有预期大小。
#wal_recycle = on                       # recycle WAL files
										# 如果设置为 on （默认值），此选项通过重命名来回收 WAL 文件，从而避免创建新文件。 
										# 在 COW 文件系统上，创建新文件系统可能更快，因此提供了禁用此行为的选项。
#wal_buffers = -1                       # min 32kB, -1 sets based on shared_buffers
                                        # (change requires restart)
										# 用于还未写入磁盘的 WAL 数据的共享内存量。
										# 默认值 -1 选择等于shared_buffers的 1/32 的尺寸（大约3%），但是不小于64kB也不大于 WAL 段的尺寸（通常为）。
										# 如果自动的选择太大或太小可以手工设置该值，但是任何小于32kB的正值都将被当作32kB。 
										# 如果指定值时没有单位，则以WAL块作为单位，即为 XLOG_BLCKSZ 字节，通常为8kB。这个参数只能在服务器启动时设置。

										# 在每次事务提交时，WAL 缓冲区的内容被写出到磁盘，因此极大的值不可能提供显著的收益。
										# 不过，把这个值设置为几个兆字节可以在一个繁忙的服务器（其中很多客户端会在同一时间提交）上提高写性能。
										# 由默认设置 -1 选择的自动调节将在大部分情况下得到合理的结果。
#wal_writer_delay = 200ms               # 1-10000 milliseconds
										# 指定 WAL 写入器刷写 WAL 的频繁程度，以时间为单位。 
										# 在刷写WAL之后，写入器将根据wal_writer_delay所给出的时间长度进行睡眠，除非被一个异步提交的事务提前唤醒。 
										# 如果最近的刷写发生在 wal_writer_delay 之前，并且小于 wal_writer_flush_after WAL的值产生之后，
										# 那么WAL只会被写入操作系统，而不会被刷写到磁盘。 
										# 如果指定值时没有单位，则以毫秒作为单位。 默
										# 认值是 200 毫秒（200ms）。
										# 注意在很多系统上，有效的睡眠延迟粒度是 10 毫秒，把wal_writer_delay设置为一个不是 10 的倍数的值，
										# 其效果和把它设置为大于该值的下一个 10 的倍数产生的效果相同。这个参数只能在postgresql.conf文件中或者服务器命令行上设置。
#wal_writer_flush_after = 1MB           # measured in pages, 0 disables
										# 指定 WAL 写入器刷写 WAL 的频繁程度，以卷为单位。 
										# 如果最近的刷写发生在 wal_writer_delay 之前，并且小于 wal_writer_flush_after WAL的值产生之后，
										# 那么WAL只会被写入操作系统，而不会被刷写到磁盘。 如果wal_writer_flush_after被设置为0，则WAL数据总是会被立即刷写。 
										# 如果指定值时没有单位，则以WAL块作为单位，即为XLOG_BLCKSZ字节，通常为8kB。 默认是1MB。
										# 这个参数只能在postgresql.conf文件中或者服务器命令行上设置。
#wal_skip_threshold = 2MB				# 当wal_level为minimal，并且在创建或重写永久关系之后提交事务时，此设置将确定如何保留新数据。 
										# 如果数据小于此设置，将其写入 WAL 日志;否则，使用受影响文件的 fsync。 
										# 根据存储的属性，如果此类提交减慢了并发事务，提高或降低此值可能会有所帮助。 
										# 如果指定此值时没有单位，则视为千字节。默认为两兆字节（2MB）。

#commit_delay = 0                       # range 0-100000, in microseconds
										# 在一次 WAL 刷写被发起之前，commit_delay增加一个时间延迟。 
										# 如果系统负载足够高，使得在一个给定间隔内有额外的事务准备好提交，
										# 那么通过允许更多事务通过一个单次 WAL 刷写来提交能够提高组提交的吞吐量。 
										# 但是，它也把每次 WAL 刷写的潜伏期增加到了最多commit_delay。 
										# 因为如果没有其他事务准备好提交，就会浪费一次延迟，只有在当一次刷写将要被发起时有至少commit_siblings个其他活动事务时，
										# 才会执行一次延迟。 另外，如果fsync被禁用，则将不会执行任何延迟。 
										# 如果指定值时没有单位，则以微秒作为单位。 
										# 默认的commit_delay是零（无延迟）。只有超级用户才能修改这个设置。

										# 在PostgreSQL的 9.3 发布之前，commit_delay的行为不同并且效果更差：
										# 它只影响提交，而不是所有 WAL 刷写，并且即使在 WAL 刷写马上就要完成时也会等待一整个配置的延迟。
										# 从PostgreSQL 9.3 中开始，第一个准备好刷写的进程会等待配置的间隔，而后续的进程只等到领先者完成刷写操作。
#commit_siblings = 5                    # range 1-1000
										# 在执行commit_delay延迟时，要求的并发活动事务的最小数目。
										# 大一些的值会导致在延迟间隔期间更可能有至少另外一个事务准备好提交。
										# 默认值是五个事务。
### - Checkpoints -  检查点相关

#checkpoint_timeout = 5min              # range 30s-1d
										# 自动 WAL 检查点之间的最长时间。如果指定值时没有单位，则以秒为单位。 合理的范围在 30 秒到 1 天之间。
										# 默认是 5 分钟（5min）。
										# 增加这个参数的值会增加崩溃恢复所需的时间。
										# 这个参数只能在postgresql.conf文件中或在服务器命令行上设置。
#checkpoint_completion_target = 0.9     # checkpoint target duration, 0.0 - 1.0
										# 指定检查点完成的目标，作为检查点之间总时间的一部分。
										# 默认是 0.5。 
										# 这个参数只能在postgresql.conf文件中或在服务器命令行上设置。
#checkpoint_flush_after = 256kB         # measured in pages, 0 disables
										# 在 Linux 和 POSIX 平台上，checkpoint_flush_after 允许强制操作系统在可配置的字节数之后将检查点写入的页面刷新到磁盘。
										# 否则，这些页面可能会保存在操作系统的页面缓存中，从而在检查点结束时发出 fsync 时导致停顿。
										# 此设置通常有助于减少事务延迟，但也会对性能产生不利影响；特别是对于大于 shared_buffers 但小于操作系统页面缓存的工作负载。
#checkpoint_warning = 30s               # 0 disables
										# 为了确保数据页的一致性，在每个检查点之后对数据页的第一次修改会导致记录整个页的内容。在这种情况下，
										# 较小的检查点间隔会增加 WAL 日志的输出量，部分否定使用较小间隔的目标，并且无论如何都会导致更多的磁盘 I/O。
										# 检查点相当昂贵，首先因为它们需要写出所有当前脏缓冲区，其次因为它们会导致额外的后续 WAL 流量，如上所述。
										# 因此，明智的做法是将检查点参数设置得足够高，这样检查点就不会经常发生。
										# 作为对检查点参数的简单完整性检查，您可以设置 checkpoint_warning 参数。
										# 如果检查点发生的时间比 checkpoint_warning 秒更近，则会向服务器日志输出一条消息，
										# 建议增加 max_wal_size。这种消息偶尔出现不会引起警报，但如果经常出现，则应增加检查点控制参数。
										# 如果您没有将 max_wal_size 设置得足够高，则诸如大型 COPY 传输之类的批量操作可能会导致出现许多此类警告。
max_wal_size = 1GB						# 服务器的检查点进程每隔一段时间就会自动执行一个检查点。
										# 每 checkpoint_timeout 秒开始一个检查点，或者如果 max_wal_size 即将被超过，
										# 以先到者为准。默认设置分别为 5 分钟和 1 GB。
										# 如果自上一个检查点以来没有写入 WAL，即使 checkpoint_timeout 已经过去，也会跳过新的检查点。 
										#（如果正在使用 WAL 归档，并且您希望对归档文件的频率设置下限以限制潜在的数据丢失，
										# 则应调整 archive_timeout 参数而不是检查点参数。）也可以强制检查点通过使用 SQL 命令 CHECKPOINT。
										
min_wal_size = 80MB						# pg_wal 目录中 WAL 段文件的数量取决于 min_wal_size、max_wal_size 和之前检查点周期中生成的 WAL 数量。
										# 当不再需要旧的日志段文件时，它们会被删除或回收（即，重命名为按编号顺序成为未来的段）。
										# 如果由于日志输出率的短期峰值超过了 max_wal_size，则将删除不需要的段文件，直到系统回到此限制之下。
										# 低于该限制，系统会回收足够的 WAL 文件来满足下一个检查点之前的估计需求，并删除其余文件。
										# 该估计值基于先前检查点周期中使用的 WAL 文件数量的移动平均值。
										# 如果实际使用量超过估计值，移动平均值会立即增加，因此它在一定程度上适应了峰值使用量而不是平均使用量。
										# min_wal_size 为将来使用的 WAL 文件的回收量设置了最小值；即使系统空闲并且 WAL 使用情况估计表明需要很少的 WAL，
										# 也总是会回收那么多 WAL 以供将来使用。

### - Archiving -  归档

archive_mode = always           # enables archiving; off, on, or always
                                # (change requires restart)
								# 启用archive_mode 后，通过设置archive_command 将完成的WAL 段发送到归档存储。
								# 除了关闭、禁用之外，还有两种模式：on和always。
								#在正常操作期间，这两种模式之间没有区别，但是当设置为 always 时，WAL 归档程序也会在归档恢复或待机模式期间启用。
								# 在always模式下，从存档中恢复的所有文件或使用流复制进行流式传输的所有文件都将被存档（再次）。有关详细信息，请参阅第 27.2.9 节。

								# archive_mode 和archive_command 是单独的变量，因此archive_command 可以在不离开归档模式的情况下进行更改。
								# 此参数只能在服务器启动时设置。
								# 当 wal_level 设置为 minimum 时，无法启用 archive_mode。
								
archive_command = '"pg_probackup" archive-push -B "/pgdata/pg_probackup" --instance pg14.2 --wal-file-name=%f --compress --archive-timeout=270'         # command to use to archive a logfile segment
                                # placeholders: %p = path of file to archive
                                #               %f = file name only
                                # e.g. 'test ! -f /mnt/server/archivedir/%f && cp %p /mnt/server/archivedir/%f'
								# 要执行以归档已完成的 WAL 文件段的本地 shell 命令。
								# 字符串中的任何 %p 都将替换为要归档的文件的路径名，而任何 %f 仅替换为文件名。 
								# 使用 %% 在命令中嵌入一个实际的 % 字符。该命令只有在成功时才返回零退出状态，这一点很重要。有关详细信息，请参阅第 26.3.1 节。

								# 此参数只能在 postgresql.conf 文件或服务器命令行中设置。
								# 除非在服务器启动时启用了 archive_mode，否则它会被忽略。
								# 如果 archive_command 是一个空字符串（默认），而 archive_mode 被启用，WAL 归档暂时被禁用，
								# 但服务器继续积累 WAL 段文件，以期很快就会提供命令。
								# 将 archive_command 设置为只返回 true 的命令，例如 /bin/true（Windows 上的 REM），有效地禁用归档，
								# 但也会破坏归档恢复所需的 WAL 文件链，因此它应该只在异常情况下使用.
archive_timeout = 900           # force a logfile segment switch after this
                                # number of seconds; 0 disables
								# archive_command 仅对已完成的 WAL 段调用。
								# 因此，如果您的服务器生成的 WAL 流量很少（或在此期间有松弛期），那么在事务完成和其在存档存储中的安全记录之间可能会有很长的延迟。
								# 要限制未归档数据的旧时间，您可以设置 archive_timeout 以强制服务器定期切换到新的 WAL 段文件。
								# 当此参数大于零时，服务器将切换到新的段文件，只要自上次段文件切换以来经过了这段时间，并且有任何数据库活动，
								# 包括单个检查点（如果存在检查点，则跳过没有数据库活动）。
								# 请注意，由于强制切换而提前关闭的存档文件的长度仍与完全完整的文件相同。
								# 因此，使用非常短的 archive_timeout 是不明智的——它会使你的存档存储膨胀。 
								# archive_timeout 1分钟左右的设置通常是合理的。
								# 如果您希望更快地将数据从主服务器复制出来，您应该考虑使用流复制而不是归档。
								# 如果指定此值没有单位，则以秒为单位。此参数只能在 postgresql.conf 文件或服务器命令行中设置。

### - Archive Recovery -

#These are only used in recovery mode.

#restore_command = ''           # command to use to restore an archived logfile segment
                                # placeholders: %p = path of file to restore
                                #               %f = file name only
                                # e.g. 'cp /mnt/server/archivedir/%f %p'
								# 要执行以检索 WAL 文件系列的存档段的本地 shell 命令。
								# 此参数对于归档恢复是必需的，但对于流复制是可选的。
								# 字符串中的任何 %p 都将替换为要归档的文件的路径名，而任何 %f 仅替换为文件名
								# （路径名是相对于当前工作目录的，即集群的数据目录。）
								# 任何 %r 都替换为包含最后一个有效重启点的文件名。
								# 这是必须保留的最早文件以允许恢复可重新启动，因此此信息可用于将存档截断为支持从当前恢复重新启动所需的最低限度。 
								# %r 通常仅用于热备份配置（请参阅第 27.2 节）。
								# 写入 %% 以嵌入实际的 % 字符。
								# restore_command = 'cp /mnt/server/archivedir/%f "%p"'
								# restore_command = 'copy "C:\\server\\archivedir\\%f" "%p"'  # Windows
								
#archive_cleanup_command = ''   # command to execute at every restartpoint
								# 此可选参数指定将在每个重新启动点执行的 shell 命令。 
								# archive_cleanup_command 的目的是提供一种机制来清理备用服务器不再需要的旧归档 WAL 文件。
								# 任何 %r 都将替换为包含最后一个有效重新启动点的文件的名称。
								# 这是必须保留的最早文件以允许可重新启动还原，因此可以安全地删除早于 %r 的所有文件。
								# 此信息可用于将存档截断为支持从当前还原重新启动所需的最低限度。 
								# pg_archivecleanup 模块经常在 archive_cleanup_command 中用于单备配置，例如：
								# archive_cleanup_command = 'pg_archivecleanup /mnt/server/archivedir %r'
								# 但是请注意，如果多个备用服务器从同一个存档目录恢复，则需要确保在任何服务器不再需要 WAL 文件之前不要删除它们。 
								# archive_cleanup_command 通常用于热备份配置（参见第 27.2 节）。
								# 写入 %% 以在命令中嵌入实际的 % 字符。
								# 如果命令返回非零退出状态，则将写入警告日志消息。
								# 此参数只能在 postgresql.conf 文件或服务器命令行中设置。
#recovery_end_command = ''      # command to execute at completion of recovery
								# 此参数指定仅在恢复结束时执行一次的 shell 命令。
								# 此参数是可选的。 
								# recovery_end_command 的目的是为复制或恢复后的清理提供一种机制。
								# 任何 %r 都将替换为包含最后一个有效重启点的文件的名称，例如在 archive_cleanup_command 中。

								# 如果命令返回非零退出状态，则将写入警告日志消息，并且数据库将继续启动。
								# 一个例外是，如果命令被 shell 的信号或错误终止（例如找不到命令），数据库将不会继续启动。

								#此参数只能在 postgresql.conf 文件或服务器命令行中设置。

### - Recovery Target -

#Set these only when performing a targeted recovery.
#默认情况下，恢复将恢复到 WAL 日志的末尾。以下参数可用于指定较早的停止点。
#最多可以使用 recovery_target、recovery_target_lsn、recovery_target_name、recovery_target_time 或 recovery_target_xid 之一；
#如果在配置文件中指定了其中一个以上，则会引发错误。
#这些参数只能在服务器启动时设置。

#recovery_target = ''           # 'immediate' to end recovery as soon as a
                                # consistent state is reached
                                # (change requires restart)
								# immediate是当前唯一允许的值。
								# 此参数指定恢复应在达到一致状态后立即结束，即尽可能早。从在线备份恢复时，这意味着备份结束的时间点。
								
#recovery_target_name = ''      # the named restore point to which recovery will proceed
                                # (change requires restart)
								# 此参数指定恢复将继续到的命名还原点（使用 pg_create_restore_point() 创建）。
#recovery_target_time = ''      # the time stamp up to which recovery will proceed
                                # (change requires restart)
								# 此参数指定将继续进行恢复的时间戳。精确停止点也受 recovery_target_inclusive 的影响。
								
								# 此参数的值是时间戳与时区数据类型所接受的格式相同的时间戳，
								# 但您不能使用时区缩写（除非在配置文件中较早地设置了 timezone_abbreviations 变量）。
								# 首选样式是使用与 UTC 的数字偏移量，或者您可以编写完整的时区名称，例如 Europe/Helsinki 而不是 EEST。
								# recovery_target_time = '2013-08-07 13:45:00+08'
								
#recovery_target_xid = ''       # the transaction ID up to which recovery will proceed
                                # (change requires restart)
								# 此参数指定恢复将继续到的事务 ID。
								# 请记住，虽然事务 ID 在事务开始时按顺序分配，但事务可以以不同的数字顺序完成。
								# 将恢复的事务是那些在指定事务之前（并且可选地包括）提交的事务。
								# 精确停止点也受 recovery_target_inclusive 的影响。
#recovery_target_lsn = ''       # the WAL LSN up to which recovery will proceed
                                # (change requires restart)
								# 此参数指定将进行恢复的预写日志位置的 LSN。
								# 精确停止点也受 recovery_target_inclusive 的影响。
								# 此参数使用系统数据类型 pg_lsn 进行解析。

#以下选项进一步指定恢复目标，并影响达到目标时发生的情况：
#recovery_target_inclusive = on # Specifies whether to stop:
                                # just after the specified recovery target (on)
                                # just before the recovery target (off)
                                # (change requires restart)
								# 指定是在指定的恢复目标之后停止 (on)，还是在恢复目标之前 (off) 停止。
								# 在指定 recovery_target_lsn、recovery_target_time 或 recovery_target_xid 时应用。
								# 此设置控制是否将具有确切目标 WAL 位置 (LSN)、提交时间或事务 ID 的事务分别包含在恢复中。
								# 默认开启。
#recovery_target_timeline = 'latest'    # 'current', 'latest', or timeline ID
                                # (change requires restart)
								# 指定恢复到特定时间线。
								# 该值可以是数字时间线 ID 或特殊值。
								# 当前值沿进行基本备份时的当前时间线恢复。
								# 值 latest 恢复到存档中找到的最新时间线，这在备用服务器中很有用。
								# latest的是默认值。
								# 通常只需要在复杂的重新恢复情况下设置此参数，需要在时间点恢复后返回到自身达到的状态。
								# 有关讨论，请参见第 26.3.5 节。
#recovery_target_action = 'pause'       # 'pause', 'promote', 'shutdown'
                                # (change requires restart)
								# 指定达到恢复目标后服务器应采取的操作。
								# 默认为暂停，这意味着恢复将被暂停。
								# 提升意味着恢复过程将完成，服务器将开始接受连接。
								# 最后关闭将在达到恢复目标后停止服务器。
								# pause设置的预期用途是允许对数据库执行查询，以检查此恢复目标是否是最理想的恢复点。
								# 可以使用 pg_wal_replay_resume() 恢复暂停状态（参见表 9.89），然后导致恢复结束。
								# 如果此恢复目标不是所需的停止点，则关闭服务器，将恢复目标设置更改为稍后的目标，然后重新启动以继续恢复。
								
								# shutdown设置对于使实例在所需的确切重放点准备好很有用。
								# 该实例仍然能够重放更多 WAL 记录（实际上，它必须在下次启动时重放自上次检查点以来的 WAL 记录）。
								
								# 请注意，因为当 recovery_target_action 设置为关闭时，recovery.signal 不会被删除，
								# 所以任何后续启动都将以立即关闭结束，除非更改配置或手动删除 recovery.signal 文件。
								
								# 如果未设置恢复目标，则此设置无效。
								# 如果未启用 hot_standby，则 pause 设置将与shutdown相同。如果在升级过程中达到恢复目标，则pause设置的作用与promote相同。

#在任何情况下，如果配置了恢复目标但存档恢复在达到目标之前结束，则服务器将关闭并出现致命错误。
								
#------------------------------------------------------------------------------
### REPLICATION
#------------------------------------------------------------------------------
#这些设置控制内置流复制功能的行为（参见第 27.2.5 节）。
#服务器将是主服务器或备用服务器。主节点可以发送数据，而备用节点始终是复制数据的接收者。
#当使用级联复制（参见第 27.2.7 节）时，备用服务器也可以是发送者，也可以是接收者。
#参数主要用于发送服务器和备用服务器，但有些参数只对主服务器有意义。
#如果需要，设置可能会因集群而异而不会出现问题。


### - Sending Servers -
#可以在将复制数据发送到一个或多个备用服务器的任何服务器上设置这些参数。
#主服务器始终是发送服务器，因此必须始终在主服务器上设置这些参数。
#这些参数的作用和含义在备用变为主后不会改变。
#Set these on the primary and on any standby that will send replication data.

#max_wal_senders = 10           # max number of walsender processes
                                # (change requires restart)
								# 指定来自备用服务器或流式基本备份客户端的最大并发连接数（即同时运行的 WAL 发送方进程的最大数量）。
								# 默认值为 10。值 0 表示禁用复制。
								# 流式客户端的突然断开可能会留下一个孤立的连接槽，直到达到超时，
								# 因此此参数应设置为略高于预期客户端的最大数量，以便断开连接的客户端可以立即重新连接。
								# 此参数只能在服务器启动时设置。
								# 此外，wal_level 必须设置为replica或更高以允许来自备用服务器的连接。

								# 运行备用服务器时，您必须将此参数设置为与主服务器上相同或更高的值。
								# 否则，备用服务器中将不允许查询。
								
#max_replication_slots = 10     # max number of replication slots
                                # (change requires restart)
								# 指定服务器可以支持的最大复制槽数（参见第 27.2.6 节）。
								# 默认值为 10。
								# 此参数只能在服务器启动时设置。
								# 将其设置为低于当前现有复制槽数的值将阻止服务器启动。
								# 此外，wal_level 必须设置为replica 或更高以允许使用复制槽。
								# 在订阅方，指定可以同时跟踪多少个复制源（参见第 50 章），有效地限制了可以在服务器上创建多少个逻辑复制订阅。
								# 将其设置为低于当前跟踪的复制源数（反映在 pg_replication_origin_status 中，而不是 pg_replication_origin）将阻止服务器启动。
								
#wal_keep_size = 0              # in megabytes; 0 disables
								# 指定保存在 pg_wal 目录中的过去日志文件段的最小大小，以防备用服务器需要获取它们以进行流复制。
								# 如果连接到发送服务器的备用服务器落后超过 wal_keep_size 兆字节，
								# 发送服务器可能会删除备用服务器仍需要的 WAL 段，在这种情况下，复制连接将被终止。
								# 结果，下游连接最终也会失败。 
								# （但是，如果 WAL 归档正在使用，备用服务器可以通过从归档中获取段来恢复。）

								# 这仅设置保留在 pg_wal 中的段的最小大小；系统可能需要为 WAL 归档或从检查点恢复保留更多段。
								# 如果 wal_keep_size 为零（默认值），系统不会为备用目的保留任何额外的段，
								# 因此备用服务器可用的旧 WAL 段的数量是前一个检查点的位置和 WAL 归档状态的函数。
								# 如果此值指定为不带单位，则以兆字节为单位。
								# 此参数只能在 postgresql.conf 文件或服务器命令行中设置。
								
#max_slot_wal_keep_size = -1    # in megabytes; -1 disables
								# 指定在检查点时复制槽允许保留在 pg_wal 目录中的 WAL 文件的最大大小。
								# 如果 max_slot_wal_keep_size 为 -1（默认值），则复制槽可以保留无限量的 WAL 文件。
								# 否则，如果复制槽的 restart_lsn 落后于当前 LSN 超过给定大小，则使用该槽的备用服务器可能由于删除所需的 WAL 文件而不再能够继续复制。
								# 您可以在 pg_replication_slots 中查看复制槽的 WAL 可用性。
								# 如果此值指定为不带单位，则以兆字节为单位。
								# 此参数只能在 postgresql.conf 文件或服务器命令行中设置。
								
#wal_sender_timeout = 60s       # in milliseconds; 0 disables
								# 终止超过此时间不活动的复制连接。
								# 这对于发送服务器检测备用崩溃或网络中断很有用。
								# 如果指定此值没有单位，则以毫秒为单位。
								# 默认值为 60 秒。
								# 零值禁用超时机制。

								# 由于集群分布在多个地理位置，每个位置使用不同的值会在集群管理中带来更大的灵活性。
								# 较小的值对于使用具有低延迟网络连接的备用数据库进行更快的故障检测很有用，
								# 而较大的值有助于更好地判断备用数据库（如果位于远程位置且具有高延迟网络连接）的运行状况。
#track_commit_timestamp = off   # collect timestamp of transaction commit
                                # (change requires restart)
								# 记录事务的提交时间。
								# 此参数只能在 postgresql.conf 文件或服务器命令行中设置。
								# 默认值为关闭。

### - Primary Server -

#These settings are ignored on a standby server.
#可以在将复制数据发送到一个或多个备用服务器的主服务器上设置这些参数。
#请注意，除了这些参数之外，还必须在主服务器上适当地设置 wal_level，并且还可以选择启用 WAL 归档（参见第 20.5.3 节）。
#备用服务器上这些参数的值无关紧要，尽管您可能希望在此处设置它们以准备备用服务器成为主服务器的可能性。
#synchronous_standby_names = '' # standby servers that provide sync rep
                                # method to choose sync standbys, number of sync standbys,
                                # and comma-separated list of application_name
                                # from standby(s); '*' = all
								# 指定可以支持同步复制的备用服务器列表，如第 27.2.8 节所述。
								# 将有一个或多个活动同步备用；在这些备用服务器确认收到其数据后，将允许等待提交的事务继续进行。
								# 同步备用数据库将是其名称出现在此列表中的那些，并且它们都是当前连接的和实时流式传输数据（
								# 如 pg_stat_replication 视图中的流式传输状态所示）。
								# 指定多个同步备用可以实现非常高的可用性并防止数据丢失。
								# 
								# 为此目的，备用服务器的名称是备用服务器的 application_name 设置，在备用服务器的连接信息中设置。
								# 在物理复制备用的情况下，这应该在 primary_conninfo 设置中设置；如果设置，则默认为 cluster_name 的设置，否则为 walreceiver。
								# 对于逻辑复制，可以在订阅的连接信息中设置，默认为订阅名。
								# 对于其他复制流消费者，请查阅他们的文档。
								# 此参数使用以下任一语法指定备用服务器列表：
								# [FIRST] num_sync ( standby_name [, ...] )
								# ANY num_sync ( standby_name [, ...] )
								# standby_name [, ...]
								# 其中 num_sync 是事务需要等待回复的同步备用服务器的数量，standby_name 是备用服务器的名称。 
								# FIRST 和 ANY 指定从列出的服务器中选择同步备用服务器的方法。
								# 关键字 FIRST 与 num_sync 结合使用，指定基于优先级的同步复制，
								# 并使事务提交等到其 WAL 记录被复制到根据优先级选择的 num_sync 同步备用数据库。
								# 名称在列表中较早出现的备用数据库具有更高的优先级，并将被视为同步的。
								# 此列表后面出现的其他备用服务器代表潜在的同步备用服务器。
								# 如果任何当前同步备用数据库由于某种原因断开连接，它将立即被下一个最高优先级备用数据库替换。
								# 关键字 FIRST 是可选的。
								

								# 第三种语法在 PostgreSQL 版本 9.6 之前使用并且仍然受支持。
								# 它与第一个语法相同，FIRST 和 num_sync 等于 1。例如，FIRST 1 (s1, s2) 和 s1, s2 具有相同的含义：选择 s1 或 s2 作为同步备用。

								# 特殊条目 * 匹配任何备用名称。

								# 没有强制备用名称唯一性的机制。在重复的情况下，匹配的备用数据库之一将被视为更高的优先级，但究竟哪一个是不确定的。
								
								# 如果此处未指定同步备用名称，则不会启用同步复制，并且事务提交不会等待复制。
								# 这是默认配置。
								# 即使启用了同步复制，也可以通过将 synchronous_commit 参数设置为 local 或 off 来将单个事务配置为不等待复制。

								# 此参数只能在 postgresql.conf 文件或服务器命令行中设置。
								
#vacuum_defer_cleanup_age = 0   # number of xacts by which cleanup is delayed
								# 指定 VACUUM 和 HOT 更新将延迟清除死行版本的事务数。
								# 默认值为零事务，这意味着可以尽快删除死行版本，也就是说，只要它们不再对任何打开的事务可见。
								# 您可能希望在支持热备用服务器的主服务器上将其设置为非零值，如第 27.4 节所述。
								# 这允许有更多时间完成备用数据库上的查询，而不会因提前清理行而引发冲突。
								# 但是，由于该值是根据主服务器上发生的写入事务数来衡量的，因此很难预测有多少额外的宽限时间可用于备用查询。
								# 此参数只能在 postgresql.conf 文件或服务器命令行中设置。

								# 您还应该考虑在备用服务器上设置 hot_standby_feedback 作为使用此参数的替代方法。

								# 这不会阻止清除已达到 old_snapshot_threshold 指定期限的死行。

### - Standby Servers -

#These settings are ignored on a primary server.
#这些设置控制接收复制数据的备用服务器的行为。它们在主服务器上的值无关紧要。


#primary_conninfo = ''                  # connection string to sending server
										# 指定用于备用服务器与发送服务器连接的连接字符串。
										# 此字符串采用第 34.1.1 节中描述的格式。
										# 如果此字符串中未指定任何选项，则检查相应的环境变量（参见第 34.15 节）。
										# 如果环境变量也未设置，则使用默认值。

										# 连接字符串应该指定发送服务器的主机名（或地址），以及端口号（如果它与备用服务器的默认值不同）。
										# 还要指定与发送服务器上具有适当特权的角色相对应的用户名（参见第 27.2.5.1 节）。
										# 如果发件人要求密码验证，也需要提供密码。
										# 它可以在 primary_conninfo 字符串中提供，也可以在备用服务器上的单独 ~/.pgpass 文件中提供（使用复制作为数据库名称）。
										# 不要在 primary_conninfo 字符串中指定数据库名称。

										# 此参数只能在 postgresql.conf 文件或服务器命令行中设置。
										# 如果在 WAL 接收器进程运行时更改了此参数，则该进程将被通知关闭并预期以新设置重新启动（除非 primary_conninfo 为空字符串）。
										# 如果服务器未处于待机模式，此设置无效。
#primary_slot_name = ''                 # replication slot on sending server
										# 可选地指定在通过流复制连接到发送服务器时要使用的现有复制槽，以控制上游节点上的资源删除（参见第 27.2.6 节）。
										# 此参数只能在 postgresql.conf 文件或服务器命令行中设置。
										# 如果在 WAL 接收器进程正在运行时更改了此参数，则该进程会发出关闭信号，并希望以新设置重新启动。
										# 如果未设置 primary_conninfo 或服务器未处于待机模式，则此设置无效。
#promote_trigger_file = ''              # file name whose presence ends recovery
										# 指定一个触发器文件，其存在结束备用数据库中的恢复。
										# 即使没有设置这个值，你仍然可以使用 pg_ctl_promote 或调用 pg_promote() 来提升备用数据库。
										# 此参数只能在 postgresql.conf 文件或服务器命令行中设置。
#hot_standby = on                       # "off" disallows queries during recovery
                                        # (change requires restart)
										# 指定您是否可以在恢复期间连接和运行查询，如第 27.4 节中所述。
										# 默认值为开启。
										# 此参数只能在服务器启动时设置。它仅在存档恢复或待机模式下有效。
										
#max_standby_archive_delay = 30s        # max delay before canceling queries
                                        # when reading WAL from archive;
                                        # -1 allows indefinite delay
										# 当 Hot Standby 处于活动状态时，此参数确定备用服务器在取消与即将应用的 WAL 条目冲突的备用查询之前应等待多长时间，
										# 如第 27.4.2 节所述。 
										# max_standby_streaming_delay 适用于通过流复制接收 WAL 数据时。
										# 如果指定此值没有单位，则以毫秒为单位。
										# 默认值为 30 秒。
										# 值 -1 允许备用数据库永远等待冲突查询完成。
										# 此参数只能在 postgresql.conf 文件或服务器命令行中设置。

										# 请注意，max_standby_archive_delay 与取消之前查询可以运行的最大时间长度不同；
										# 相反，它是允许应用任何一个 WAL 段数据的最大总时间。
										# 因此，如果一个查询在 WAL 段的早期导致了显着延迟，那么后续的冲突查询将具有更少的宽限时间。
										
#max_standby_streaming_delay = 30s      # max delay before canceling queries
                                        # when reading streaming WAL;
                                        # -1 allows indefinite delay
										# 当 Hot Standby 处于活动状态时，此参数确定备用服务器在取消与即将应用的 WAL 条目冲突的备用查询之前应等待多长时间，
										# 如第 27.4.2 节所述。 
										# max_standby_streaming_delay 适用于通过流复制接收 WAL 数据时。
										# 如果指定此值没有单位，则以毫秒为单位。
										# 默认值为 30 秒。
										# 值 -1 允许备用数据库永远等待冲突查询完成。
										# 此参数只能在 postgresql.conf 文件或服务器命令行中设置。

										# 请注意，max_standby_streaming_delay 与取消之前查询可以运行的最大时间长度不同；
										# 相反，它是从主服务器接收到 WAL 数据后允许应用的最大总时间。
										# 因此，如果一个查询导致了明显的延迟，那么后续的冲突查询将有更少的宽限时间，直到备用服务器再次赶上。
										
#wal_receiver_create_temp_slot = off    # create temp slot if primary_slot_name
                                        # is not set
										# 指定当没有配置要使用的永久复制槽时（使用 primary_slot_name），WAL 接收器进程是否应在远程实例上创建临时复制槽。
										# 默认为关闭。
										# 此参数只能在 postgresql.conf 文件或服务器命令行中设置。
										# 如果在 WAL 接收器进程正在运行时更改了此参数，则该进程会发出关闭信号，并希望以新设置重新启动。
#wal_receiver_status_interval = 10s     # send replies at least this often
                                        # 0 disables
										# 指定备用服务器上的 WAL 接收器进程向主备用服务器或上游备用服务器发送有关复制进度的信息的最小频率，
										# 可以使用 pg_stat_replication 视图查看。
										# 备用数据库将报告它写入的最后一个预写日志位置、它刷新到磁盘的最后一个位置以及它应用的最后一个位置。
										# 此参数的值是报告之间的最长时间。
										# 每次写入或刷新位置更改时发送更新，或者如果设置为非零值，则按照此参数指定的频率发送更新。
										# 还有其他情况会在忽略此参数的情况下发送更新；
										# 例如，当现有 WAL 的处理完成或 synchronous_commit 设置为 remote_apply 时。
										# 因此，应用位置可能会稍微落后于真实位置。
										# 如果指定此值没有单位，则以秒为单位。
										# 默认值为 10 秒。
										# 此参数只能在 postgresql.conf 文件或服务器命令行中设置。
#hot_standby_feedback = off             # send info from standby to prevent
                                        # query conflicts
										# 指定热备库是否将有关当前在备库上执行的查询的反馈发送到主库或上游备库。
										# 此参数可用于消除因清除记录而导致的查询取消，但可能会导致某些工作负载的主数据库膨胀。
										# 每个 wal_receiver_status_interval 发送反馈消息的频率不会超过一次。
										# 默认值为关闭。
										# 此参数只能在 postgresql.conf 文件或服务器命令行中设置。

										# 如果级联复制正在使用，则反馈将向上传递，直到最终到达主节点。除了向上游传递之外，备用服务器不使用他们收到的反馈。

										# 此设置不会覆盖主节点上 old_snapshot_threshold 的行为；
										# 备用数据库上超过主数据库年龄阈值的快照可能会变得无效，从而导致备用数据库上的事务取消。
										# 这是因为 old_snapshot_threshold 旨在为死行可能导致膨胀的时间提供绝对限制，否则会因为备用配置而被违反。
										
#wal_receiver_timeout = 60s             # time that receiver waits for
                                        # communication from primary
                                        # in milliseconds; 0 disables
										# 终止超过此时间不活动的复制连接。
										# 这对于接收备用服务器检测主节点崩溃或网络中断很有用。
										# 如果指定此值没有单位，则以毫秒为单位。
										# 默认值为 60 秒。
										# 零值禁用超时机制。
										# 此参数只能在 postgresql.conf 文件或服务器命令行中设置。
#wal_retrieve_retry_interval = 5s       # time to wait before retrying to
                                        # retrieve WAL after a failed attempt
										# 指定当 WAL 数据不能从任何源（流复制、本地 pg_wal 或 WAL 归档）获得时，备用服务器应等待多长时间，
										# 然后再尝试检索 WAL 数据。
										# 如果指定此值没有单位，则以毫秒为单位。
										# 默认值为 5 秒。
										# 此参数只能在 postgresql.conf 文件或服务器命令行中设置。

										# 此参数在恢复中的节点需要控制等待新 WAL 数据可用的时间量的配置中很有用。
										# 例如，在归档恢复中，可以通过减小该参数的值来使恢复在检测到新的 WAL 日志文件时更加灵敏。
										# 在 WAL 活动较低的系统上，增加它会减少访问 WAL 档案所需的请求数量，这在考虑访问基础设施的次数的云环境中很有用。
										
#recovery_min_apply_delay = 0           # minimum delay for applying changes during recovery
										# 默认情况下，备用服务器会尽快从发送服务器恢复 WAL 记录。
										# 拥有数据的延时副本可能很有用，它提供了纠正数据丢失错误的机会。
										# 此参数允许您将恢复延迟指定的时间量。
										# 例如，如果您将此参数设置为 5min，则仅当备用数据库上的系统时间比主数据库报告的提交时间至少 5 分钟时，
										# 备用数据库才会重播每个事务提交。
										# 如果指定此值没有单位，则以毫秒为单位。
										# 默认为零，不添加延迟。

### - Subscribers -
#这些设置控制逻辑复制订阅者的行为。
#请注意，wal_receiver_timeout、wal_receiver_status_interval 和 wal_retrieve_retry_interval 配置参数也会影响逻辑复制工作者。
#These settings are ignored on a publisher.

#max_logical_replication_workers = 4    # taken from max_worker_processes
                                        # (change requires restart)
										# 指定逻辑复制工作者的最大数量。这包括应用工作者和表同步工作者。

										# 逻辑复制工作者从 max_worker_processes 定义的池中获取。

										# 默认值为 4。
										# 此参数只能在服务器启动时设置。
#max_sync_workers_per_subscription = 2  # taken from max_logical_replication_workers
										# 每个订阅的最大同步工作者数。此参数控制订阅初始化期间或添加新表时初始数据副本的并行度。

										# 目前，每个表只能有一个同步工作者。

										# 同步工作者从 max_logical_replication_workers 定义的池中获取。

										# 默认值为 2。
										# 该参数只能在 postgresql.conf 文件或服务器命令行中设置。


#------------------------------------------------------------------------------
## QUERY TUNING
#------------------------------------------------------------------------------

### - Planner Method Configuration -
#这些配置参数提供了一种影响查询优化器选择的查询计划的粗略方法。
#如果优化器为特定查询选择的默认计划不是最优的，临时解决方案是使用这些配置参数之一来强制优化器选择不同的计划。
#提高优化器选择的计划质量的更好方法包括调整计划器成本常量（参见第 20.7.2 节）、手动运行 ANALYZE、增加 default_statistics_target 配置参数的值
#以及增加为特定列收集的统计信息量使用 ALTER TABLE SET STATISTICS。

#enable_async_append = on 					# 启用或禁用查询计划器对异步感知追加计划类型的使用。默认为开启。
#enable_bitmapscan = on 					# 启用或禁用查询计划器对位图扫描计划类型的使用。默认为开启。
#enable_gathermerge = on 					# 启用或禁用查询计划器对收集合并计划类型的使用。默认为开启。
#enable_hashagg = on  						# 启用或禁用查询计划器对散列聚合计划类型的使用。默认为开启。
#enable_hashjoin = on						# 启用或禁用查询计划器对散列连接计划类型的使用。默认为开启。
#enable_incremental_sort = on				# 启用或禁用查询计划器对增量排序步骤的使用。默认为开启。
#enable_indexscan = on						# 启用或禁用查询计划器对索引扫描计划类型的使用。默认为开启。
#enable_indexonlyscan = on					# 启用或禁用查询计划器对仅索引扫描计划类型的使用（请参阅第 11.9 节）。默认为开启。
#enable_material = on						# 启用或禁用查询计划器对物化的使用。完全抑制物化是不可能的，但是关闭这个变量可以防止规划器插入物化节点，除非在需要正确性的情况下。默认为开启。
#enable_memoize = on						# 启用或禁用查询计划器使用 memoize 计划缓存来自嵌套循环连接内的参数化扫描的结果。当当前参数的结果已经在缓存中时，此计划类型允许跳过对基础计划的扫描。当新条目需要更多空间时，可能会从缓存中逐出不太常见的查找结果。默认为开启。
#enable_mergejoin = on						# 启用或禁用查询计划器对合并连接计划类型的使用。默认为开启。
#enable_nestloop = on						# 启用或禁用查询计划器对嵌套循环连接计划的使用。完全禁止嵌套循环连接是不可能的，但是如果有其他方法可用，关闭这个变量会阻止计划器使用一个。默认为开启。
#enable_parallel_append = on				# 启用或禁用查询计划器对并行感知追加计划类型的使用。默认为开启。
#enable_parallel_hash = on					# 启用或禁用查询计划器使用具有并行散列的散列连接计划类型。如果散列连接计划也未启用，则无效。默认为开启。
#enable_partition_pruning = on				# 启用或禁用查询计划器从查询计划中消除分区表的分区的能力。这也控制了规划器生成查询计划的能力，这些查询计划允许查询执行器在查询执行期间删除（忽略）分区。默认为开启。有关详细信息，请参阅第 5.11.4 节。
#enable_partitionwise_join = off			# 启用或禁用查询计划器对分区联接的使用，这允许通过联接匹配分区来执行分区表之间的联接。 Partitionwise join 当前仅适用于连接条件包括所有分区键的情况，这些分区键必须是相同的数据类型并且具有一对一匹配的子分区集。因为分区连接计划在计划期间可以使用显着更多的 CPU 时间和内存，所以默认是关闭的。
#enable_partitionwise_aggregate = off		# 启用或禁用查询计划器对分区分组或聚合的使用，这允许对每个分区单独执行的分区表进行分组或聚合。如果 GROUP BY 子句不包含分区键，则只能在每个分区的基础上执行部分聚合，并且必须稍后执行最终确定。因为按分区分组或聚合可以在规划期间使用显着更多的 CPU 时间和内存，所以默认设置为关闭。
#enable_seqscan = on                        # 启用或禁用查询计划程序对顺序扫描计划类型的使用。完全禁止顺序扫描是不可能的，但是如果有其他可用的方法，关闭这个变量会阻止规划器使用一个。默认为开启。
#enable_sort = on                           # 启用或禁用查询计划器对显式排序步骤的使用。完全禁止显式排序是不可能的，但是如果有其他可用的方法，关闭这个变量会阻止规划器使用一个。默认为开启。
#enable_tidscan = on                        # 启用或禁用查询计划程序对 TID 扫描计划类型的使用。默认为开启。

### - Planner Cost Constants -计划员成本常数
#本节中描述的成本变量是在任意尺度上测量的。只有它们的相对值很重要，因此将它们全部放大或缩小相同的因子不会导致规划者的选择发生变化。
#默认情况下，这些成本变量基于顺序页面获取的成本；也就是说，seq_page_cost 通常设置为 1.0，其他成本变量参考该值设置。
#但是，如果您愿意，可以使用不同的比例，例如在特定机器上以毫秒为单位的实际执行时间。

#注意
#不幸的是，没有明确的方法来确定成本变量的理想值。
#最好将它们视为特定安装将收到的整个查询组合的平均值。
#这意味着仅在几次实验的基础上更改它们是非常冒险的。

#seq_page_cost = 1.0                    # measured on an arbitrary scale 
										#设置计划者对作为一系列顺序提取的一部分的磁盘页面提取成本的估计。
										# 默认值为 1.0。
										# 通过设置同名的表空间参数，可以为特定表空间中的表和索引覆盖此值（请参阅 ALTER TABLESPACE）。
#random_page_cost = 4.0                 # same scale as above
										# 设置计划者对非顺序获取磁盘页面成本的估计。
										# 默认值为 4.0。通过设置同名的表空间参数，可以为特定表空间中的表和索引覆盖此值（请参阅 ALTER TABLESPACE）。
										# 相对于 seq_page_cost 减小这个值会导致系统更喜欢索引扫描；提高它会使索引扫描看起来相对更昂贵。
										# 可以同时提高或降低这两个值，以更改磁盘 I/O 成本相对于 CPU 成本的重要性，由以下参数描述。
										# 随机访问机械磁盘存储通常比顺序访问昂贵四倍多。
										# 但是，使用较低的默认值 (4.0)，因为大多数对磁盘的随机访问，例如索引读取，都假定在缓存中。
										# 可以将默认值视为将随机访问建模为比顺序访问慢 40 倍，同时期望 90% 的随机读取被缓存。
										
										# 如果您认为 90% 的缓存率对于您的工作负载来说是不正确的假设，您可以增加 random_page_cost 以更好地反映随机
										# 存储读取的真实成本。
										# 相应地，如果您的数据可能完全在缓存中，例如当数据库小于服务器总内存时，减少 random_page_cost 可能是合适的。
										# 相对于顺序读取成本较低的存储，例如固态驱动器，也可以使用较低的 random_page_cost 值（例如 1.1）更好地建模。
#提示
#尽管系统会让您将 random_page_cost 设置为小于 seq_page_cost，但这样做在物理上并不明智。
#但是，如果数据库完全缓存在 RAM 中，则将它们设置为相等是有意义的，因为在这种情况下，不按顺序触摸页面不会受到惩罚。
#此外，在一个高速缓存的数据库中，应该相对于 CPU 参数降低这两个值，因为获取已经在 RAM 中的页面的成本比通常要小得多。

#cpu_tuple_cost = 0.01                  # same scale as above 设置计划者对查询期间处理每一行的成本的估计。默认值为 0.01。
#cpu_index_tuple_cost = 0.005           # same scale as above 设置计划者对索引扫描期间处理每个索引条目的成本的估计。默认值为 0.005。
#cpu_operator_cost = 0.0025             # same scale as above 设置计划者对处理查询期间执行的每个运算符或函数的成本的估计。默认值为 0.0025。
#parallel_setup_cost = 1000.0   # same scale as above 设置计划者对启动并行工作进程成本的估计。默认值为 1000。
#parallel_tuple_cost = 0.1              # same scale as above 设置计划者对将一个元组从并行工作进程转移到另一个进程的成本的估计。默认值为 0.1。
#min_parallel_table_scan_size = 8MB		# 设置为考虑并行扫描而必须扫描的最小表数据量。
										# 对于并行顺序扫描，扫描的表数据量总是等于表的大小，但是当使用索引时，扫描的表数据量通常会更少。
										# 如果这个值没有指定单位，它被视为块，即 BLCKSZ 字节，通常为 8kB。
										# 默认值为 8 兆字节 (8MB)。
#min_parallel_index_scan_size = 512kB	# 设置为考虑并行扫描而必须扫描的最小索引数据量。
										# 请注意，并行索引扫描通常不会触及整个索引。
										# 规划者认为扫描实际会触及的页数是相关的。
										# 该参数还用于决定特定索引是否可以参与并行vacuum。见vacuum。
										# 如果这个值没有指定单位，它被视为块，即 BLCKSZ 字节，通常为 8kB。
										# 默认值为 512 KB (512kB)。
										
#effective_cache_size = 4GB             # 文件系统缓存，通常设置为总内存80% 50%~90%  多 少 并不会有严重影响
										# 设置规划器对单个查询可用的磁盘缓存的有效大小的假设。
										# 这被计入使用指数的成本估算中；较高的值使其更有可能使用索引扫描，较低的值使其更有可能使用顺序扫描。
										# 设置此参数时，您应该同时考虑 PostgreSQL 的共享缓冲区和将用于 PostgreSQL 数据文件的内核磁盘缓存部分，
										# 尽管这两个地方可能都存在一些数据。此外，还要考虑不同表上的预期并发查询数，因为它们必须共享可用空间。
										# 该参数对PostgreSQL分配的共享内存大小没有影响，也不保留内核磁盘缓存；它仅用于估计目的。
										# 系统也不假设数据在查询之间保留在磁盘缓存中。
										# 如果这个值没有指定单位，它被视为块，即 BLCKSZ 字节，通常为 8kB。
										# 默认值为 4 GB (4GB)。 （如果 BLCKSZ 不是 8kB，默认值会按比例缩放。）
										
#jit_above_cost = 100000                # perform JIT compilation if available
                                        # and query more expensive than this;
                                        # -1 disables
										# 如果启用，则设置激活 JIT 编译的查询成本（参见第 32 章）。
										# 执行 JIT 会花费计划时间，但可以加速查询执行。
										# 将此设置为 -1 将禁用 JIT 编译。
										# 默认值为 100000。
#jit_inline_above_cost = 500000         # inline small functions if query is
                                        # more expensive than this; -1 disables
										# 设置 JIT 编译尝试内联函数和运算符的查询成本。内联增加了计划时间，但可以提高执行速度。
										# 将此设置为小于 jit_above_cost 没有意义。
										# 将此设置为 -1 将禁用内联。
										# 默认值为 500000。
#jit_optimize_above_cost = 500000       # use expensive JIT optimizations if
                                        # query is more expensive than this;
                                        # -1 disables
										# 设置 JIT 编译应用昂贵优化的查询成本。这种优化增加了计划时间，但可以提高执行速度。
										# 将其设置为小于 jit_above_cost 没有意义，将其设置为大于 jit_inline_above_cost 不太可能有益。
										# 将此设置为 -1 会禁用昂贵的优化。
										# 默认值为 500000。

### - Genetic Query Optimizer - 查询优化器
#遗传查询优化器 (GEQO) 是一种使用启发式搜索进行查询规划的算法。
#这减少了复杂查询（加入许多关系的查询）的计划时间，代价是生成的计划有时不如正常的穷举搜索算法找到的计划。
#有关详细信息，请参阅第 60 章。

#geqo = on
										# 启用或禁用遗传查询优化。这是默认开启的。通常最好不要在生产中关闭它； geqo_threshold 变量提供对 GEQO 的更精细控制。
#geqo_threshold = 12					# 使用遗传查询优化来计划至少涉及这么多 FROM 项的查询。 （
										# 请注意，FULL OUTER JOIN 构造仅计为一个 FROM 项。）
										# 默认值为 12。
										# 对于更简单的查询，通常最好使用常规的详尽搜索计划器，但对于具有许多表的查询，
										# 详尽搜索需要太长时间，通常比执行次优计划的惩罚要长。
										# 因此，查询大小的阈值是管理 GEQO 使用的便捷方式。
#geqo_effort = 5                        # range 1-10
										# 控制 GEQO 中计划时间和查询计划质量之间的权衡。
										# 此变量必须是 1 到 10 范围内的整数。
										# 默认值为 5。
										# 较大的值会增加执行查询计划所花费的时间，但也会增加选择有效查询计划的可能性。
										
#geqo_pool_size = 0                     # selects default based on effort
										# 控制 GEQO 使用的池大小，即遗传种群中的个体数量。
										# 它必须至少为两个，有用的值通常是 100 到 1000。
										# 如果将其设置为零（默认设置），则根据 geqo_effort 和查询中的表数选择合适的值。
#geqo_generations = 0                   # selects default based on effort
										# 控制 GEQO 使用的代数，即算法的迭代次数。
										# 它必须至少为 1，并且有用的值与池大小在同一范围内。
										# 如果它设置为零（默认设置），则根据 geqo_pool_size 选择合适的值。
#geqo_selection_bias = 2.0              # range 1.5-2.0
										# 控制 GEQO 使用的选择偏差。选择偏差是群体内的选择压力。值可以从 1.50 到 2.00；后者是默认值。
#geqo_seed = 0.0                        # range 0.0-1.0
										# 控制 GEQO 用于通过连接顺序搜索空间选择随机路径的随机数生成器的初始值。
										# 该值的范围可以从零（默认值）到一。
										# 改变值会改变探索的连接路径集，并可能导致找到更好或更差的最佳路径。

### - Other Planner Options -  其他规划器选项

#default_statistics_target = 100        # range 1-10000
										# 为没有通过 ALTER TABLE SET STATISTICS 设置特定列目标的表列设置默认统计目标。
										# 较大的值会增加执行 ANALYZE 所需的时间，但可能会提高规划者估计的质量。
										# 默认值为 100。
										# 有关 PostgreSQL 查询计划器使用统计信息的更多信息，请参阅第 14.2 节。
#constraint_exclusion = partition       # on, off, or partition
										# 控制查询计划器使用表约束来优化查询。 
										# constraint_exclusion 的允许值为 on（检查所有表的约束）、off（从不检查约束）和 partition（仅检查继承子表和 UNION ALL 子查询的约束）。
										# 分区是默认设置。
										# 它通常与传统的继承树一起使用以提高性能。

										# 当此参数允许它用于特定表时，计划程序将查询条件与表的 CHECK 约束进行比较，并省略扫描条件与约束相矛盾的表。
										# 启用约束排除后，此 SELECT 将根本不扫描 child1000，从而提高性能。
										# 目前，默认情况下，仅对经常用于通过继承树实现表分区的情况启用约束排除。
										# 为所有表打开它会带来额外的计划开销，这在简单查询中非常明显，并且通常不会对简单查询产生任何好处。
										# 如果您没有使用传统继承进行分区的表，您可能更愿意完全关闭它。 
										# （请注意，分区表的等效功能由单独的参数 enable_partition_pruning 控制。）
#cursor_tuple_fraction = 0.1            # range 0.0-1.0
										# 设置计划者对将被检索的游标行的分数的估计。
										# 默认值为 0.1。
										# 此设置的较小值会使规划器倾向于使用游标的“快速启动”规划，这将快速检索前几行，而可能需要很长时间才能获取所有行。
										# 较大的值更加强调总估计时间。
										# 在最大设置为 1.0 时，游标的计划与常规查询完全一样，只考虑总估计时间，而不考虑第一行的交付时间。
#from_collapse_limit = 8
										# 如果生成的 FROM 列表不超过这么多项目，则计划器会将子查询合并到上层查询中。
										# 较小的值会减少计划时间，但可能会产生较差的查询计划。
										# 默认值为八。
										# 有关详细信息，请参阅第 14.3 节。
#jit = on                               # allow JIT compilation 
										# 确定 PostgreSQL 是否可以使用 JIT 编译（如果可用）（参见第 32 章）。默认为开启。
#join_collapse_limit = 8                # 1 disables collapsing of explicit
                                        # JOIN clauses
										# 每当生成不超过这么多项目的列表时，计划器将把显式 JOIN 构造（FULL JOIN 除外）重写为 FROM 项目列表。
										# 较小的值会减少计划时间，但可能会产生较差的查询计划。
										# 默认情况下，此变量设置为与 from_collapse_limit 相同，适用于大多数用途。
										# 将其设置为 1 可防止显式 JOIN 的任何重新排序。
										# 因此，查询中指定的显式连接顺序将是连接关系的实际顺序。
										# 因为查询计划器并不总是选择最佳连接顺序，高级用户可以选择暂时将此变量设置为 1，然后明确指定他们想要的连接顺序。
										# 有关详细信息，请参阅第 14.3 节。
										# 将此值设置为 geqo_threshold 或更大可能会触发 GEQO 计划器的使用，从而导致非最佳计划。请参阅第 20.7.3 节。
#plan_cache_mode = auto                 # auto, force_generic_plan or
                                        # force_custom_plan
										# 准备好的语句（显式准备或隐式生成，例如通过 PL/pgSQL）可以使用自定义或通用计划执行。
										# 使用其特定的参数值集为每次执行重新制定自定义计划，而通用计划不依赖于参数值并且可以跨执行重复使用。
										# 因此，使用通用计划可以节省计划时间，但如果理想计划强烈依赖于参数值，那么通用计划可能效率低下。
										# 这些选项之间的选择通常是自动进行的，但可以用 plan_cache_mode 覆盖。
										# 允许的值为 auto（默认值）、force_custom_plan 和 force_generic_plan。
										# 当要执行缓存计划时会考虑此设置，而不是在准备好计划时。有关更多信息，请参阅准备。


#------------------------------------------------------------------------------
## REPORTING AND LOGGING
#------------------------------------------------------------------------------

### - Where to Log -

log_destination = 'stderr'              # Valid values are combinations of
                                        # stderr, csvlog, syslog, and eventlog,
                                        # depending on platform.  csvlog
                                        # requires logging_collector to be on.
										# PostgreSQL 支持多种记录服务器消息的方法，包括 stderr、csvlog 和 syslog。在 Windows 上，还支持事件日志。
										# 将此参数设置为以逗号分隔的所需日志目标列表。
										# 默认情况下仅记录到 stderr。
										# 此参数只能在 postgresql.conf 文件或服务器命令行中设置。

										# 如果log_destination中包含csvlog，则日志条目以“逗号分隔值”（CSV）格式输出，便于将日志加载到程序中。
										# 必须启用 logging_collector 才能生成 CSV 格式的日志输出。

										# current_logfiles 在创建新日志文件作为轮换的效果时重新创建，并且当重新加载 log_destination 时。
										# 当 log_destination 中既不包含 stderr 也不包含 csvlog 并且禁用了日志收集器时，它将被删除。

#This is used when logging to stderr:
logging_collector = on          # Enable capturing of stderr and csvlog
                                        # into log files. Required to be on for
                                        # csvlogs.
                                        # (change requires restart)
										# 此参数启用日志收集器，这是一个后台进程，用于捕获发送到 stderr 的日志消息并将它们重定向到日志文件中。
										# 这种方法通常比记录到 syslog 更有用，因为某些类型的消息可能不会出现在 syslog 输出中。 
										# （一个常见的例子是动态链接器失败消息；另一个是脚本产生的错误消息，例如archive_command。）
										# 这个参数只能在服务器启动时设置。

#These are only used if logging_collector is on:
log_directory = 'log'                   # directory where log files are written,
                                        # can be absolute or relative to PGDATA
										# 启用 logging_collector 时，此参数确定将在其中创建日志文件的目录。
										# 它可以指定为绝对路径，或相对于集群数据目录。
										# 此参数只能在 postgresql.conf 文件或服务器命令行中设置。
										# 默认为日志。
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log' # log file name pattern,
                                        # can include strftime() escapes
log_file_mode = 0600                    # creation mode for log files,
                                        # begin with 0 to use octal notation
										# 参数值应为以 chmod 和 umask 系统调用接受的格式指定的数字模式。 
										# （要使用惯用的八进制格式，数字必须以 0（零）开头。）
log_rotation_age = 1d                   # Automatic rotation of logfiles will
                                        # happen after that time.  0 disables.
										# 启用 logging_collector 时，此参数确定使用单个日志文件的最长时间，之后将创建一个新的日志文件。
										# 如果此值指定为不带单位，则以分钟为单位。
										# 默认值为 24 小时。
										# 设置为零以禁用基于时间的新日志文件创建。
										# 此参数只能在 postgresql.conf 文件或服务器命令行中设置。
log_rotation_size = 10MB                # Automatic rotation of logfiles will
                                        # happen after that much log output.
                                        # 0 disables.
										# 此参数确定单个日志文件的最大大小。
										# 在将这些数据量发送到日志文件后，将创建一个新的日志文件。
										# 如果此值指定为不带单位，则以千字节为单位。
										# 默认值为 10 兆字节。
										# 设置为零以禁用基于大小的新日志文件创建。
										# 此参数只能在 postgresql.conf 文件或服务器命令行中设置。
#log_truncate_on_rotation = off         # If on, an existing log file with the
                                        # same name as the new log file will be
                                        # truncated rather than appended to.
                                        # But such truncation only occurs on
                                        # time-driven rotation, not on restarts
                                        # or size-driven rotation.  Default is
                                        # off, meaning append to existing files
                                        # in all cases.
										# 此参数将导致 PostgreSQL 截断（覆盖）而不是附加到任何现有的同名日志文件。
										# 但是，仅当由于基于时间的轮换而打开新文件时才会发生截断，而不是在服务器启动或基于大小的轮换期间。
										# 关闭时，在所有情况下都将附加预先存在的文件。
										# 例如，将此设置与 postgresql-%H.log 之类的 log_filename 结合使用将导致生成 24 小时日志文件，然后循环覆盖它们。
										# 此参数只能在 postgresql.conf 文件或服务器命令行中设置。
										
										# 示例：要保留7天的日志，每天一个日志文件名为server_log.Mon、server_log.Tue等，并自动用本周的日志覆盖上周的日志，
										# 设置：
										# log_filename = server_log.%a
										# log_truncate_on_rotation = on
										# log_rotation_age = 1440

#These are relevant when logging to syslog:
#syslog_facility = 'LOCAL0'
										# 启用日志记录到 syslog 时，此参数确定要使用的 syslog “工具”。
										# 您可以选择LOCAL0、LOCAL1、LOCAL2、LOCAL3、LOCAL4、LOCAL5、LOCAL6、LOCAL7；
										# 默认为 LOCAL0。另请参阅系统的 syslog 守护程序的文档。
										# 此参数只能在 postgresql.conf 文件或服务器命令行中设置。
#syslog_ident = 'postgres'				# 启用日志记录到 syslog 时，此参数确定用于标识 syslog 日志中的 PostgreSQL 消息的程序名称。
										# 默认为 postgres。
										# 此参数只能在 postgresql.conf 文件或服务器命令行中设置。
#syslog_sequence_numbers = on			# 当登录到 syslog 并且这是打开的（默认设置）时，每条消息都将以递增的序列号作为前缀（例如 [2]）。
										# 这绕过了许多 syslog 实现默认执行的“---最后一条消息重复 N 次 ---”抑制。
										# 在更现代的 syslog 实现中，可以配置重复消息抑制（例如，rsyslog 中的 $RepeatedMsgReduction），因此这可能不是必需的。
										# 此外，如果您真的想抑制重复的消息，您可以将其关闭。
#syslog_split_messages = on				# 启用日志记录到 syslog 时，此参数确定如何将消息传递到 syslog。
										# 启用时（默认），消息按行拆分，长行被拆分，以便它们适合 1024 字节，这是传统 syslog 实现的典型大小限制。
										# 关闭时，PostgreSQL 服务器日志消息按原样传递给 syslog 服务，由 syslog 服务来处理可能庞大的消息。

										# 如果 syslog 最终记录到一个文本文件，那么任何一种方式的效果都是一样的，最好保持设置打开，
										# 因为大多数 syslog 实现要么无法处理大消息，要么需要专门配置来处理它们。
										# 但是，如果 syslog 最终要写入其他介质，则将消息逻辑地放在一起可能是必要的或更有用的。

										# 此参数只能在 postgresql.conf 文件或服务器命令行中设置。

#This is only relevant when logging to eventlog (Windows):
#(change requires restart)
#event_source = 'PostgreSQL'
										# 启用记录到事件日志时，此参数确定用于标识日志中的 PostgreSQL 消息的程序名称。
										# 默认为 PostgreSQL。
										# 此参数只能在 postgresql.conf 文件或服务器命令行中设置。

### - When to Log -

#log_min_messages = warning             # values in order of decreasing detail:
                                        #   debug5
                                        #   debug4
                                        #   debug3
                                        #   debug2
                                        #   debug1
                                        #   info
                                        #   notice
                                        #   warning
                                        #   error
                                        #   log
                                        #   fatal
                                        #   panic
										# 控制将哪些消息级别写入服务器日志。
										# 每个级别包括其后的所有级别。
										# 级别越高，发送到日志的消息就越少。
										# 默认值为警告。
										# 请注意，此处的 LOG 与 client_min_messages 中的排名不同。只有超级用户可以更改此设置。
										# 
										
#log_min_error_statement = error        # values in order of decreasing detail:
                                        #   debug5
                                        #   debug4
                                        #   debug3
                                        #   debug2
                                        #   debug1
                                        #   info
                                        #   notice
                                        #   warning
                                        #   error
                                        #   log
                                        #   fatal
                                        #   panic (effectively off)
										# 控制在服务器日志中记录哪些导致错误条件的 SQL 语句。
										# 当前 SQL 语句包含在指定严重性或更高严重性的任何消息的日志条目中。
										# 默认值为 ERROR，这意味着将记录导致错误、日志消息、致命错误或恐慌的语句。
										# 要有效关闭失败语句的日志记录，请将此参数设置为 PANIC。
										# 只有超级用户可以更改此设置。
										
#log_min_duration_statement = -1        # -1 is disabled, 0 logs all statements
                                        # and their durations, > 0 logs only
                                        # statements running at least this number
                                        # of milliseconds
										# 如果语句至少运行了指定的时间，则记录每个已完成语句的持续时间。
										# 例如，如果您将其设置为 250 毫秒，那么所有运行 250 毫秒或更长时间的 SQL 语句都将被记录。
										# 启用此参数有助于跟踪应用程序中未优化的查询。
										# 如果指定此值没有单位，则以毫秒为单位。将此设置为零会打印所有语句持续时间。 
										# -1（默认值）禁用记录语句持续时间。只有超级用户可以更改此设置。

										# 这将覆盖 log_min_duration_sample，这意味着持续时间超过此设置的查询不受采样的影响，并且始终被记录。

										# 对于使用扩展查询协议的客户端，Parse、Bind 和 Execute 步骤的持续时间是独立记录的。
										# 当将此选项与 log_statement 一起使用时，由于 log_statement 而记录的语句文本将不会在持续时间日志消息中重复。
										# 如果您不使用 syslog，建议您使用 log_line_prefix 记录 PID 或会话 ID，
										# 以便您可以使用进程 ID 或会话 ID 将语句消息链接到稍后的持续时间消息。

#log_min_duration_sample = -1           # -1 is disabled, 0 logs a sample of statements
                                        # and their durations, > 0 logs only a sample of
                                        # statements running at least this number
                                        # of milliseconds;
                                        # sample fraction is determined by log_statement_sample_rate
										# 允许对至少运行指定时间量的已完成语句的持续时间进行采样。
										# 这会产生与 log_min_duration_statement 相同类型的日志条目，但仅针对已执行语句的子集，
										# 采样率由 log_statement_sample_rate 控制。
										# 例如，如果将其设置为 100 毫秒，那么所有运行 100 毫秒或更长的 SQL 语句都将被考虑进行采样。
										# 当流量太高而无法记录所有查询时，启用此参数会很有帮助。
										# 如果指定此值没有单位，则以毫秒为单位。
										# 将此设置为零对所有语句持续时间进行采样。 
										# -1（默认值）禁用采样语句持续时间。
										# 只有超级用户可以更改此设置。

										# 此设置的优先级低于 log_min_duration_statement，这意味着持续时间超过 log_min_duration_statement 的语句不受采样并始终记录。

#log_statement_sample_rate = 1.0        # fraction of logged statements exceeding
                                        # log_min_duration_sample to be logged;
                                        # 1.0 logs all such statements, 0.0 never logs
										# 确定将记录的持续时间超过 log_min_duration_sample 的语句的比例。
										# 抽样是随机的，例如 0.5 意味着在统计上，有二分之一的机会记录任何给定的语句。
										# 默认值为 1.0，表示记录所有采样语句。
										# 将此设置为零将禁用采样语句持续时间日志记录，与将 log_min_duration_sample 设置为 -1 相同。
										# 只有超级用户可以更改此设置。


#log_transaction_sample_rate = 0.0      # fraction of transactions whose statements
                                        # are logged regardless of their duration; 1.0 logs all
                                        # statements from all transactions, 0.0 never logs
										# 除了出于其他原因记录的语句之外，设置其语句全部记录的事务的比例。
										# 它适用于每笔新交易，无论其语句的持续时间如何。
										# 抽样是随机的，例如 0.1 表示统计上有十分之一的机会记录任何给定的交易。 
										# log_transaction_sample_rate 有助于构建交易样本。
										# 默认值为 0，表示不记录来自任何其他事务的语句。
										# 将此设置为 1 记录所有事务的所有语句。
										# 只有超级用户可以更改此设置。
										
										# 与所有语句记录选项一样，此选项会增加大量开销。
### - What to Log -

#application_name = 
										# application_name 可以是少于 NAMEDATALEN 字符的任何字符串（标准构建中为 64 个字符）。
										# 它通常由应用程序在连接到服务器时设置。
										# 该名称将显示在 pg_stat_activity 视图中并包含在 CSV 日志条目中。
										# 它也可以通过 log_line_prefix 参数包含在常规日志条目中。 
										# application_name 值中只能使用可打印的 ASCII 字符。其他字符将替换为问号 (?)。
#debug_print_parse = off
#debug_print_rewritten = off
#debug_print_plan = off
										# 这些参数可以发出各种调试输出。
										# 设置后，它们会打印生成的解析树、查询重写器输出或每个已执行查询的执行计划。
										# 这些消息在 LOG 消息级别发出，因此默认情况下它们会出现在服务器日志中，但不会发送到客户端。
										# 您可以通过调整 client_min_messages 和/或 log_min_messages 来更改它。
										# 这些参数默认是关闭的。
#debug_pretty_print = on
										# 设置后，debug_pretty_print 缩进由 debug_print_parse、debug_print_rewritten 或 debug_print_plan 生成的消息。
										# 与关闭时使用的“紧凑”格式相比，这会产生更具可读性但更长的输出。
										# 默认开启。
#log_autovacuum_min_duration = -1       # log autovacuum activity;
                                        # -1 disables, 0 logs all actions and
                                        # their durations, > 0 logs only
                                        # actions running at least this number
                                        # of milliseconds.
										# 如果 autovacuum 执行的每个操作至少运行了指定的时间，则记录该操作。
										# 将此设置为零会记录所有 autovacuum 操作。 
										# -1（默认值）禁用记录 autovacuum 操作。
										# 如果指定此值没有单位，则以毫秒为单位。
										# 例如，如果您将其设置为 250 毫秒，那么所有运行 250 毫秒或更长时间的autovacuum和分析都会被记录下来。
										# 此外，当此参数设置为 -1 以外的任何值时，如果由于锁冲突或同时删除的关系而跳过自动清理操作，则会记录一条消息。
										# 启用此参数有助于跟踪 autovacuum 活动。
										# 该参数只能在 postgresql.conf 文件或服务器命令行中设置；但是可以通过更改表存储参数来覆盖单个表的设置。
#log_checkpoints = off
										# 导致在服务器日志中记录检查点和重新启动点。
										# 日志消息中包含一些统计信息，包括写入的缓冲区数量和写入它们所花费的时间。
										# 此参数只能在 postgresql.conf 文件或服务器命令行中设置。
										# 默认为关闭。
										
#log_connections = off
										# 导致记录到服务器的每个尝试连接，以及成功完成客户端身份验证（如果需要）和授权。
										# 只有超级用户可以在会话开始时更改此参数，并且在会话中根本无法更改。
										# 默认为关闭。
										# 一些客户端程序，如 psql，在确定是否需要密码时尝试连接两次，因此重复的“收到连接”消息并不一定表示存在问题。
#log_disconnections = off				
										# 导致记录会话终止。
										# 日志输出提供类似于 log_connections 的信息，以及会话的持续时间。
										# 只有超级用户可以在会话开始时更改此参数，并且在会话中根本无法更改。
										# 默认为关闭。
#log_duration = off						# 导致记录每个已完成语句的持续时间。
										# 默认为关闭。
										# 只有超级用户可以更改此设置。

										# 对于使用扩展查询协议的客户端，Parse、Bind 和 Execute 步骤的持续时间是独立记录的。
										
#log_error_verbosity = default          # terse, default, or verbose messages
										# 控制记录在服务器日志中的每条消息的详细信息量。
										# 有效值为 TERSE、DEFAULT 和 VERBOSE，每个都向显示的消息添加更多字段。 
										# TERSE 不包括 DETAIL、HINT、QUERY 和 CONTEXT 错误信息的记录。 
										# VERBOSE 输出包括 SQLSTATE 错误代码（另请参阅附录 A）以及生成错误的源代码文件名、函数名和行号。
										# 只有超级用户可以更改此设置。
#log_hostname = off						# 
										# 默认情况下，连接日志消息仅显示连接主机的 IP 地址。
										# 打开此参数也会导致记录主机名。
										# 请注意，根据您的主机名解析设置，这可能会带来不可忽略的性能损失。
										# 此参数只能在 postgresql.conf 文件或服务器命令行中设置。
#log_line_prefix = '%m [%p] '           # special values:
                                        #   %a = application name
                                        #   %u = user name
                                        #   %d = database name
                                        #   %r = remote host and port
                                        #   %h = remote host
                                        #   %b = backend type
                                        #   %p = process ID
                                        #   %P = process ID of parallel group leader
                                        #   %t = timestamp without milliseconds
                                        #   %m = timestamp with milliseconds
                                        #   %n = timestamp with milliseconds (as a Unix epoch)
                                        #   %Q = query ID (0 if none or not computed)
                                        #   %i = command tag
                                        #   %e = SQL state
                                        #   %c = session ID
                                        #   %l = session line number
                                        #   %s = session start timestamp
                                        #   %v = virtual transaction ID
                                        #   %x = transaction ID (0 if none)
                                        #   %q = stop here in non-session
                                        #        processes
                                        #   %% = '%'
                                        # e.g. '<%u%%%d> '
#这是一个 printf 样式的字符串，在每个日志行的开头输出。 
#% 字符开始“转义序列”，这些字符被替换为如下所述的状态信息。
#无法识别的转义将被忽略。其他字符直接复制到日志行。
#有些转义只有会话进程才能识别，会被主服务器进程等后台进程视为空。
#通过在 % 之后和选项之前指定数字文字，状态信息可以左对齐或右对齐。
#负值将导致状态信息在右侧填充空格以使其具有最小宽度，而正值将在左侧填充。
#填充有助于提高日志文件的人类可读性。

#此参数只能在 postgresql.conf 文件或服务器命令行中设置。
#默认值为 '%m [%p] '，它记录时间戳和进程 ID。
#后端类型对应于视图 pg_stat_activity 中的列 backend_type，但在该视图中未显示的日志中可能会出现其他类型。

#%c 转义打印一个准唯一的会话标识符，由两个 4 字节的十六进制数字（不带前导零）组成，
#由一个点分隔。数字是进程开始时间和进程 ID，因此 %c 也可以用作打印这些项目的节省空间的方式。
#例如，要从 pg_stat_activity 生成会话标识符，请使用以下查询：
#提示
#如果您为 log_line_prefix 设置了一个非空值，您通常应该将其最后一个字符设置为空格，以提供与日志行其余部分的视觉分离。也可以使用标点符号。
登录后复制
提示
									# Syslog 会生成自己的时间戳和进程 ID 信息，因此如果您正在登录到 syslog，您可能不想包含这些转义。
									# 提示
									# 当包含仅在会话（后端）上下文中可用的信息（如用户或数据库名称）时，%q 转义很有用。例如：
#log_line_prefix = '%m [%p] %q%u@%d/%a ’
# 笔记
# %Q 转义总是为 log_statement 输出的行报告零标识符，因为 log_statement 在可以计算标识符之前生成输出，包括无法计算标识符的无效语句。

#log_lock_waits = off # log lock waits >= deadlock_timeout
# 控制是否生成日志消息,当会话等待比 deadlock_timeout 更长的时间来获取锁时。
# 这对于确定锁定等待是否导致性能不佳很有用。
# 默认为关闭。
# 只有超级用户可以更改此设置。

#log_recovery_conflict_waits = off # log standby recovery conflict waits
# >= deadlock_timeout
# 控制是否生成日志消息,当启动进程等待恢复冲突的时间超过 deadlock_timeout 时。
# 这在确定恢复冲突是否会阻止恢复应用 WAL 时很有用。
#log_parameter_max_length = -1 # when logging statements, limit logged
# bind-parameter values to N bytes;
# -1 means print in full, 0 disables
# 如果大于零，则使用非错误语句记录消息记录的每个绑定参数值将被修剪到这么多字节。
# 零禁用非错误语句日志的绑定参数的记录。
# -1（默认值）允许完整记录绑定参数。
# 如果此值指定为不带单位，则将其视为字节。
# 只有超级用户可以更改此设置。

									# 此设置仅影响由于 log_statement、log_duration 和相关设置而打印的日志消息。此
									# 设置的非零值会增加一些开销，特别是如果以二进制形式发送参数，则需要转换为文本。

#log_parameter_max_length_on_error = 0  # when logging an error, limit logged
                                        # bind-parameter values to N bytes;
                                        # -1 means print in full, 0 disables
#如果大于零，则错误消息中报告的每个绑定参数值都会被修剪到这么多字节。
#零（默认值）禁用在错误消息中包含绑定参数。 
#-1 允许完整打印绑定参数。
#如果此值指定为不带单位，则将其视为字节。
#log_statement = 'none'                 # none, ddl, mod, all
#控制记录哪些 SQL 语句。
#有效值为 none (off)、ddl、mod 和 all（所有语句）。 
#ddl 记录所有数据定义语句，例如 CREATE、ALTER 和 DROP 语句。 
#mod 记录所有 ddl 语句，以及数据修改语句，例如 INSERT、UPDATE、DELETE、TRUNCATE 和 COPY FROM。
#如果 PREPARE、EXECUTE 和 EXPLAIN ANALYZE 语句包含的命令属于适当的类型，则它们也会被记录。
#对于使用扩展查询协议的客户端，当接收到 Execute 消息时发生日志记录，并且包含 Bind 参数的值（任何嵌入的单引号加倍）。

#默认值为无。只有超级用户可以更改此设置。
#即使通过 log_statement = all 设置，也不会记录包含简单语法错误的语句，
#因为只有在完成基本解析以确定语句类型后才会发出日志消息。
#在扩展查询协议的情况下，此设置同样不会记录在执行阶段之前失败的语句（即，在解析分析或规划期间）。
#将 log_min_error_statement 设置为 ERROR（或更低）以记录此类语句。
#log_replication_commands = off
#使每个复制命令都记录在服务器日志中。
#有关复制命令的更多信息，请参阅第 53.4 节。
										# 默认值为关闭。只有超级用户可以更改此设置。
#log_temp_files = -1                    # log temporary files equal or larger
                                        # than the specified size in kilobytes;
                                        # -1 disables, 0 logs all temp files
										# 控制临时文件名和大小的记录。可以为排序、散列和临时查询结果创建临时文件。
										# 如果通过此设置启用，则在删除每个临时文件时都会发出一个日志条目。
										# 零值记录所有临时文件信息，而正值仅记录大小大于或等于指定数据量的文件。
										# 如果此值指定为不带单位，则以千字节为单位。
										# 默认设置为 -1，即禁用此类日志记录。
#只有超级用户可以更改此设置。
log_timezone = 'PRC'
#设置用于写入服务器日志的时间戳的时区。
#与 TimeZone 不同，此值是集群范围的，因此所有会话将一致地报告时间戳。
#内置默认值是 GMT，但通常在 postgresql.conf 中被覆盖； initdb 将在那里安装与其系统环境相对应的设置。
#有关详细信息，请参阅第 8.5.3 节。
#此参数只能在 postgresql.conf 文件或服务器命令行中设置。

#------------------------------------------------------------------------------
## PROCESS TITLE

#这些设置控制如何修改服务器进程的进程标题。
#进程标题通常使用 ps 或 Windows 上的 Process Explorer 等程序查看。
#有关详细信息，请参阅第 28.1 节。
#------------------------------------------------------------------------------

#cluster_name = ''                      # added to process titles if nonempty
										# (change requires restart)
										# 为各种目的设置一个标识此数据库集群（实例）的名称。
										# 集群名称出现在该集群中所有服务器进程的进程标题中。
										# 此外，它是备用连接的默认应用程序名称（请参阅 synchronous_standby_names。）
#update_process_title = on
										# 每次服务器接收到新的 SQL 命令时启用更新进程标题。
										# 此设置在大多数平台上默认为打开，但在 Windows 上默认为关闭，因为该平台更新进程标题的开销较大。
										# 只有超级用户可以更改此设置。

#------------------------------------------------------------------------------
## STATISTICS
#------------------------------------------------------------------------------

### - Query and Index Statistics Collector -查询和索引统计收集器
#这些参数控制服务器范围的统计信息收集功能。
#启用统计信息收集后，可以通过系统视图的 pg_stat 和 pg_statio 系列访问生成的数据。
#track_activities = on
										# 启用有关每个会话的当前执行命令的信息的收集，以及它的标识符和该命令开始执行的时间。
										# 该参数默认开启。
										# 请注意，即使启用，此信息也不对所有用户可见，仅对超级用户和拥有正在报告的会话的用户可见，因此它不应代表安全风险。
										# 只有超级用户可以更改此设置。
#track_activity_query_size = 1024       # (change requires restart)
										# 为 pg_stat_activity.query 字段指定为存储每个活动会话的当前执行命令的文本而保留的内存量。
#如果此值指定为不带单位，则将其视为字节。
										# 默认值为 1024 字节。
										# 此参数只能在服务器启动时设置。
#track_counts = on						# 启用有关数据库活动的统计信息的收集。
										# 该参数默认开启，因为 autovacuum 守护进程需要收集的信息。
										# 只有超级用户可以更改此设置。
#track_io_timing = off					# 启用数据库 I/O 调用的计时。
										# 该参数默认关闭，因为它会重复查询操作系统的当前时间，这可能会在某些平台上造成很大的开销。
										# 您可以使用 pg_test_timing 工具来测量系统上的计时开销。 
										# I/O 计时信息显示在 pg_stat_database 中，当使用 BUFFERS 选项时，
										# 在 EXPLAIN 的输出中，由 autovacuum 用于自动清理和自动分析，当设置 log_autovacuum_min_duration 时
										# 以及由 pg_stat_statements 显示。
						
				# 只有超级用户可以更改此设置。
#track_wal_io_timing = off				# 启用 WAL I/O 调用的计时。
										# 该参数默认关闭，因为它会重复查询操作系统的当前时间，这可能会在某些平台上造成很大的开销。
										# 您可以使用 pg_test_timing 工具来测量系统上的计时开销。
										# I/O 时序信息显示在 pg_stat_wal 中。
										# 只有超级用户可以更改此设置。
#track_functions = none                 # none, pl, all
										# 启用对函数调用计数和使用时间的跟踪。
										# 指定 pl 仅跟踪过程语言函数，all 也跟踪 SQL 和 C 语言函数。
										# 默认值为 none，即禁用函数统计跟踪。只有超级用户可以更改此设置。
										
										# 无论此设置如何，都不会跟踪简单到可以“内联”到调用查询中的 SQL 语言函数。
#stats_temp_directory = 'pg_stat_tmp'
										# 设置存储临时统计数据的目录。
										# 这可以是相对于数据目录的路径或绝对路径。
										# 默认值为 pg_stat_tmp。
										# 将其指向基于 RAM 的文件系统将降低物理 I/O 需求并提高性能。
										# 此参数只能在 postgresql.conf 文件或服务器命令行中设置。

### - Monitoring - 统计监控


#compute_query_id = auto
										# 启用查询标识符的核心计算。
										# 查询标识符可以使用 EXPLAIN 显示在 pg_stat_activity 视图中，或者如果通过 log_line_prefix 参数配置，则在日志中发出。
										# pg_stat_statements 扩展还需要计算一个查询标识符。
										# 请注意，如果内核查询标识符计算方法不可接受，则可以选择使用外部模块。
										# 在这种情况下，必须始终禁用内核计算。
										# 有效值是 off（总是禁用）、on（总是启用）和 auto，这让 pg_stat_statements 等模块自动启用它。
										# 默认为auto。
										
										# 为了确保只计算和显示一个查询标识符，如果已经计算了一个查询标识符，计算查询标识符的扩展程序应该抛出一个错误。
#log_statement_stats = off
#log_parser_stats = off
#log_planner_stats = off
#log_executor_stats = off
										# 对于每个查询，将相应模块的性能统计信息输出到服务器日志。
										# 这是一个粗略的分析工具，类似于 Unix getrusage() 操作系统工具。 
										# log_statement_stats 报告总语句统计信息，而其他报告每个模块的统计信息。 
										# log_statement_stats 不能与任一模块选项一起启用。
										# 默认情况下，所有这些选项都被禁用。
										# 只有超级用户可以更改这些设置。

#------------------------------------------------------------------------------
## AUTOVACUUM
#这些设置控制 autovacuum 功能的行为。有关详细信息，请参阅第 25.1.6 节。
#请注意，其中许多设置可以在每个表的基础上被覆盖；请参阅存储参数。
#------------------------------------------------------------------------------

#autovacuum = on                        # Enable autovacuum subprocess?  'on'
                                        # requires track_counts to also be on.
										# 控制服务器是否应该运行 autovacuum 启动器守护程序。
										# 这是默认开启的；但是，还必须启用 track_counts 才能使 autovacuum 工作。
										# 该参数只能在 postgresql.conf 文件或服务器命令行中设置；
										# 但是，可以通过更改表存储参数来禁用单个表的自动清理。

										# 请注意，即使禁用此参数，系统也会在必要时启动 autovacuum 进程以防止事务 ID 回绕。
										# 有关详细信息，请参阅第 25.1.5 节。
										
#autovacuum_max_workers = 3             # max number of autovacuum subprocesses
                                        # (change requires restart)
										# 指定可以在任何时候运行的最大自动清理进程数（除了自动清理启动程序）。
										# 默认值为三个。此参数只能在服务器启动时设置。
#autovacuum_naptime = 1min              # time between autovacuum runs
										# 指定在任何给定数据库上运行 autovacuum 之间的最小延迟。
										# 在每一轮中，守护程序检查数据库并根据需要为该数据库中的表发出 VACUUM 和 ANALYZE 命令。
										# 如果指定此值没有单位，则以秒为单位。
										# 默认值为一分钟 (1min)。此参数只能在 postgresql.conf 文件或服务器命令行中设置。
#autovacuum_vacuum_threshold = 50       # min number of row updates before
                                        # vacuum
										# 指定在任何一个表中触发 VACUUM 所需的最小更新或删除元组数。
										# 默认值为 50 个元组。
										# 该参数只能在 postgresql.conf 文件或服务器命令行中设置；但是可以通过更改表存储参数来覆盖单个表的设置。
#autovacuum_vacuum_insert_threshold = 1000      # min number of row inserts
                                        # before vacuum; -1 disables insert
                                        # vacuums
										# 指定在任何一个表中触发 VACUUM 所需的插入元组数。
										# 默认值为 1000 个元组。
										# 如果指定了 -1，autovacuum 将不会根据插入次数对任何表触发 VACUUM 操作。
										# 该参数只能在 postgresql.conf 文件或服务器命令行中设置；
										# 但是可以通过更改表存储参数来覆盖单个表的设置。
#autovacuum_analyze_threshold = 50      # min number of row updates before
                                        # analyze
										# 指定在任何一个表中触发 ANALYZE 所需的最小插入、更新或删除元组数。
										# 默认值为 50 个元组。
										# 该参数只能在 postgresql.conf 文件或服务器命令行中设置；但是可以通过更改表存储参数来覆盖单个表的设置。
#autovacuum_vacuum_scale_factor = 0.2   # fraction of table size before vacuum
										# 在决定是否触发 VACUUM 时，指定要添加到 autovacuum_vacuum_threshold 的表大小的一部分。
										# 默认值为 0.2（表大小的 20%）。
										# 该参数只能在 postgresql.conf 文件或服务器命令行中设置；但是可以通过更改表存储参数来覆盖单个表的设置。
#autovacuum_vacuum_insert_scale_factor = 0.2    # fraction of inserts over table
                                        # size before insert vacuum
										# 在决定是否触发 VACUUM 时，指定要添加到 autovacuum_vacuum_insert_threshold 的表大小的一部分。
										# 默认值为 0.2（表大小的 20%）。
										# 该参数只能在 postgresql.conf 文件或服务器命令行中设置；但是可以通过更改表存储参数来覆盖单个表的设置。
#autovacuum_analyze_scale_factor = 0.1  # fraction of table size before analyze
										# 在决定是否触发 ANALYZE 时，指定要添加到 autovacuum_analyze_threshold 的表大小的一部分。
										# 默认值为 0.1（表大小的 10%）。
										# 该参数只能在 postgresql.conf 文件或服务器命令行中设置；但是可以通过更改表存储参数来覆盖单个表的设置。
#autovacuum_freeze_max_age = 200000000  # maximum XID age before forced vacuum
                                        # (change requires restart)
										# 指定表的 pg_class.relfrozenxid 字段在强制执行 VACUUM 操作以防止表内的事务 ID 回绕之前可以达到的最大年龄（在事务中）。
										# 请注意，即使在 autovacuum 以其他方式禁用时，系统也会启动 autovacuum 进程以防止回绕。

										# Vacuum 还允许从 pg_xact 子目录中删除旧文件，这就是为什么默认是相对较低的 2 亿事务。
										# 该参数只能在服务器启动时设置，但可以通过更改表存储参数来减少单个表的设置。
										# 有关详细信息，请参阅第 25.1.5 节。
#autovacuum_multixact_freeze_max_age = 400000000        # maximum multixact age
                                        # before forced vacuum
                                        # (change requires restart)
										# 指定表的 pg_class.relminmxid 字段在强制执行 VACUUM 操作以防止表内的 multixact ID 环绕之前可以达到的最大年龄（在 multixact 中）。
										# 请注意，即使在 autovacuum 以其他方式禁用时，系统也会启动 autovacuum 进程以防止回绕。

										# 清理 multixacts 还允许从 pg_multixact/members 和 pg_multixact/offsets 子目录中删除旧文件，
										# 这就是为什么默认是相对较低的 4 亿 multixacts。
										# 该参数只能在服务器启动时设置，但可以通过更改表存储参数来减少单个表的设置。
										# 有关详细信息，请参阅第 25.1.5.1 节。
#autovacuum_vacuum_cost_delay = 2ms     # default vacuum cost delay for
                                        # autovacuum, in milliseconds;
                                        # -1 means use vacuum_cost_delay
										# 指定将在自动 VACUUM 操作中使用的成本延迟值。
										# 如果指定了 -1，将使用常规的 Vacuum_cost_delay 值。
										# 如果指定此值没有单位，则以毫秒为单位。
										# 默认值为 2 毫秒。
										# 该参数只能在 postgresql.conf 文件或服务器命令行中设置；但是可以通过更改表存储参数来覆盖单个表的设置。
#autovacuum_vacuum_cost_limit = -1      # default vacuum cost limit for
                                        # autovacuum, -1 means use
                                        # vacuum_cost_limit
										# 指定将在自动 VACUUM 操作中使用的成本限制值。
										# 如果指定了 -1（这是默认值），则将使用常规的 Vacuum_cost_limit 值。
										# 请注意，该值会按比例分配给正在运行的 autovacuum 工作人员（如果有多个工作人员），
										# 因此每个工作人员的限制总和不会超过此变量的值。
										# 该参数只能在 postgresql.conf 文件或服务器命令行中设置；但是可以通过更改表存储参数来覆盖单个表的设置。


#------------------------------------------------------------------------------
## CLIENT CONNECTION DEFAULTS 客户端连接默认值
#------------------------------------------------------------------------------

### - Statement Behavior -

#client_min_messages = notice           # values in order of decreasing detail: 按value的顺序减少细节
                                        #   debug5
                                        #   debug4
                                        #   debug3
                                        #   debug2
                                        #   debug1
                                        #   log
                                        #   notice
                                        #   warning
                                        #   error
										# 控制将哪些消息级别发送到客户端。
										# 有效值为 DEBUG5、DEBUG4、DEBUG3、DEBUG2、DEBUG1、LOG、NOTICE、WARNING 和 ERROR。
										# 每个级别包括其后的所有级别。
										# 级别越高，发送的消息就越少。默认值为通知。
										# 请注意，这里的 LOG 与 log_min_messages 中的排名不同。

										# INFO 级别的消息总是发送给客户端。
										
#search_path = '"$user", public'        # schema names
										# 此变量指定在未指定模式的简单名称引用对象（表、数据类型、函数等）时搜索模式的顺序。
										# 当不同模式中存在同名对象时，使用在搜索路径中首先找到的对象。
										# 不在搜索路径中的任何模式中的对象只能通过使用限定（点分）名称指定其包含模式来引用。
										# search_path 的值必须是以逗号分隔的模式名称列表。
										# 任何不是现有架构的名称，或者是用户没有 USAGE 权限的架构，都会被静默忽略。
										
										# 如果列表项之一是特殊名称 $user，则替换具有由 CURRENT_USER 返回的名称的模式，如果存在这样的模式并且用户对其具有 USAGE 权限。 
										# （如果不是，$user 将被忽略。）
										# 系统目录模式 pg_catalog 总是被搜索，无论它是否在路径中被提及。
										# 如果它在路径中被提及，那么它将按照指定的顺序进行搜索。
										# 如果 pg_catalog 不在路径中，则将在搜索任何路径项之前对其进行搜索。
										
										# 同样，则始终搜索当前会话的临时表模式 pg_temp_nnn，如果它存在。
										# 它可以通过使用别名 pg_temp 在路径中显式列出。
										# 如果它没有在路径中列出，则首先搜索它（甚至在 pg_catalog 之前）。
										# 但是，临时模式仅搜索关系（表、视图、序列等）和数据类型名称。它永远不会搜索函数或运算符名称。
										
										# 当对象在没有指定特定目标模式的情况下创建时，它们将被放置在 search_path 中命名的第一个有效模式中。
										# 如果搜索路径为空，则会报错。
										
										# 此参数的默认值为“$user”，public。
										# 此设置支持共享使用数据库（其中没有用户具有私有模式，并且所有共享使用公共）、私有的每用户模式以及这些的组合。
										# 通过更改默认搜索路径设置（全局或每个用户）可以获得其他效果。
										
										# 有关模式处理的更多信息，请参阅第 5.9 节。
										# 特别是，默认配置仅适用于数据库只有一个用户或几个相互信任的用户时。
										
										# 可以通过 SQL 函数 current_schemas 检查搜索路径的当前有效值（参见第 9.26 节）。
										# 这与检查 search_path 的值不太一样，因为 current_schemas 显示了出现在 search_path 中的项目是如何被解析的。
#row_security = on						# 此变量控制是否引发错误以代替应用行安全策略。
										# 设置为 on 时，策略将正常应用。
										# 当设置为关闭时，查询会失败，否则至少会应用一项策略。
										# 默认为开启。
										# 在有限的行可见性可能导致错误结果的情况下更改为关闭；
										# 例如， pg_dump 默认情况下会进行该更改。
										# 此变量对绕过每行安全策略的角色、超级用户和具有 BYPASSRLS 属性的角色没有影响。
										
										# 有关行安全策略的更多信息，请参阅 CREATE POLICY。
										
#default_table_access_method = 'heap'	# 如果 CREATE 命令未显式指定访问方法，或者使用 SELECT ... INTO 时，
										# 该参数指定创建表或实体化视图时使用的默认表访问方法，该方法不允许指定表访问方法。
										# 默认是heap。
#default_tablespace = ''                # a tablespace name, '' uses the default
										# 当 CREATE 命令未显式指定表空间时，此变量指定在其中创建对象（表和索引）的默认表空间。
										
										# 该值可以是表空间的名称，也可以是指定使用当前数据库的默认表空间的空字符串。
										# 如果该值与任何现有表空间的名称不匹配，PostgreSQL 将自动使用当前数据库的默认表空间。
										# 如果指定了非默认表空间，则用户必须对其具有 CREATE 权限，否则创建尝试将失败。
										# 该变量不用于临时表；对他们来说，改为咨询 temp_tablespaces。
										# 创建数据库时也不使用此变量。
										# 默认情况下，新数据库从复制它的模板数据库继承其表空间设置。
										# 如果在创建分区表时将该参数设置为空字符串以外的值，则该分区表的表空间将设置为该值，
										# 该值将用作将来创建的分区的默认表空间，即使 default_tablespace 有从此改变。

										# 有关表空间的更多信息，请参阅第 23.6 节。
										
#default_toast_compression = 'pglz'     # 'pglz' or 'lz4'
										# 此变量为可压缩列的值设置默认的 TOAST 压缩方法。 
										# （这可以通过在 CREATE TABLE 或 ALTER TABLE 中设置 COMPRESSION 列选项来覆盖单个列。）
										# 支持的压缩方法是 pglz 和（如果 PostgreSQL 是用 --with-lz4 编译的）lz4。
										# 默认为 pglz。
#temp_tablespaces = ''                  # a list of tablespace names, '' uses
                                        # only default tablespace
										# 当 CREATE 命令未显式指定表空间时，此变量指定要在其中创建临时对象（临时表和临时表上的索引）的表空间。
										# 在这些表空间中还会创建用于对大型数据集进行排序等目的的临时文件。

										# 该值是表空间名称的列表。
										# 当列表中有多个名称时，PostgreSQL 会在每次创建临时对象时随机选择列表中的一个成员；
										# 除了在事务中，连续创建的临时对象被放置在列表中的连续表空间中。
										# 如果列表的选定元素是空字符串，PostgreSQL 将自动使用当前数据库的默认表空间。

										# 当 temp_tablespaces 以交互方式设置时，指定不存在的表空间是错误的，指定用户没有 CREATE 权限的表空间也是错误的。
										# 但是，当使用先前设置的值时，不存在的表空间将被忽略，用户缺乏 CREATE 权限的表空间也是如此。
										# 特别是，当使用 postgresql.conf 中设置的值时，此规则适用。

										# 默认值为空字符串，这会导致在当前数据库的默认表空间中创建所有临时对象。

										# 另请参见 default_tablespace。
#check_function_bodies = on				# 该参数常开。
										# 当设置为 off 时，它会在 CREATE FUNCTION 和 CREATE PROCEDURE 期间禁用例程正文字符串的验证。
										# 禁用验证可以避免验证过程的副作用，特别是防止由于前向引用等问题导致的误报。
										# 在代表其他用户加载功能之前将此参数设置为关闭； pg_dump 会自动执行此操作。
#default_transaction_isolation = 'read committed'
										# 每个 SQL 事务都有一个隔离级别，可以是“未提交读”、“已提交读”、“可重复读”或“可序列化”。
										# 此参数控制每个新事务的默认隔离级别。
										# 默认值为“已提交读”。

										# 有关更多信息，请参阅第 13 章和 SET TRANSACTION。
										
#default_transaction_read_only = off	# 只读 SQL 事务不能更改非临时表。
										# 此参数控制每个新事务的默认只读状态。
										# 默认为关闭（读/写）。

										# 有关更多信息，请参阅 SET TRANSACTION。
										
#default_transaction_deferrable = off	# 在可序列化隔离级别上运行时，可延迟的只读 SQL 事务可能会在允许继续之前被延迟。
										# 然而，一旦它开始执行，它就不会产生任何确保可串行化所需的开销；
										# 因此序列化代码没有理由因为并发更新而强制中止它，这使得该选项适用于长时间运行的只读事务。

										# 此参数控制每个新事务的默认可延迟状态。
										# 它目前对读写事务或以低于可序列化的隔离级别运行的事务没有影响。
										# 默认为关闭。

										# 有关更多信息，请参阅 SET TRANSACTION。
#session_replication_role = 'origin'	# 控制当前会话的复制相关触发器和规则的触发。
										# 设置此变量需要超级用户权限，并导致丢弃任何以前缓存的查询计划。
										# 可能的值是 origin（默认值）、replica 和 local。

										# 此设置的预期用途是逻辑复制系统在应用复制更改时将其设置为副本。
										# 这样做的效果是触发器和规则（尚未从其默认配置更改）不会在副本上触发。
										# 有关详细信息，请参阅 ALTER TABLE 子句 ENABLE TRIGGER 和 ENABLE RULE。

										# PostgreSQL 在内部将设置 origin 和 local 视为相同。
										# 第三方复制系统可能会出于内部目的使用这两个值，例如使用 local 来指定不应复制其更改的会话。

										# 由于外键是作为触发器实现的，因此将此参数设置为副本也会禁用所有外键检查，如果使用不当，可能会使数据处于不一致状态。
#statement_timeout = 0                  # in milliseconds, 0 is disabled
										# 中止任何花费超过指定时间的语句。
										# 如果 log_min_error_statement 设置为 ERROR 或更低，超时的语句也将被记录。
										# 如果指定此值没有单位，则以毫秒为单位。
										# 零值（默认值）禁用超时。

										# 超时时间是从命令到达服务器到服务器完成的时间。
										# 如果多个 SQL 语句出现在单个 simple-Query 消息中，则超时将分别应用于每个语句。 
										# （PostgreSQL 13 之前的版本通常将超时视为应用于整个查询字符串。）
										# 在扩展查询协议中，超时在任何与查询相关的消息（Parse、Bind、Execute、Describe）到达时开始运行，
										# 并在完成时取消执行或同步消息。

										# 不建议在 postgresql.conf 中设置 statement_timeout，因为它会影响所有会话。
										
#lock_timeout = 0                       # in milliseconds, 0 is disabled
										# 在尝试获取表、索引、行或其他数据库对象上的锁时，中止等待超过指定时间量的任何语句。
										# 时间限制分别适用于每次锁定获取尝试。
										# 该限制适用于显式锁定请求（例如 LOCK TABLE 或 SELECT FOR UPDATE without NOWAIT）和隐式获取的锁。
										# 如果指定此值没有单位，则以毫秒为单位。
										# 零值（默认值）禁用超时。

										# 与 statement_timeout 不同，此超时只能在等待锁时发生。
										# 请注意，如果 statement_timeout 不为零，则将 lock_timeout 设置为相同或更大的值是毫无意义的，因为语句超时总是首先触发。
										# 如果 log_min_error_statement 设置为 ERROR 或更低，将记录超时的语句。

										# 不建议在 postgresql.conf 中设置 lock_timeout，因为它会影响所有会话。
										
#idle_in_transaction_session_timeout = 0        # in milliseconds, 0 is disabled
										# 终止在打开的事务中空闲（即等待客户端查询）超过指定时间量的任何会话。
										# 如果指定此值没有单位，则以毫秒为单位。
										# 零值（默认值）禁用超时。

										# 此选项可用于确保空闲会话不会在不合理的时间内保持锁定。
										# 即使没有持有重要的锁，打开的事务也可以防止清除可能仅对该事务可见的最近死亡的元组；
										# 所以长时间保持空闲会导致表膨胀。有关详细信息，请参阅第 25.1 节。

#idle_session_timeout = 0               # in milliseconds, 0 is disabled
										# 终止任何已空闲（即等待客户端查询）但不在打开的事务中超过指定时间量的会话。
										# 如果指定此值没有单位，则以毫秒为单位。
										# 零值（默认值）禁用超时。

										# 与打开事务的情况不同，没有事务的空闲会话不会给服务器带来很大的开销，因此与 idle_in_transaction_session_timeout 相比，
										# 启用此超时的需要更少。

										# 小心对通过连接池软件或其他中间件建立的连接强制执行此超时，因为这样的层可能对意外的连接关闭反应不佳。
										# 仅对交互式会话启用此超时可能会有所帮助，也许仅将其应用于特定用户。
										
#vacuum_freeze_table_age = 150000000
										# 如果表的 pg_class.relfrozenxid 字段已达到此设置指定的年龄，VACUUM 将执行积极扫描。
										# 主动扫描与常规 VACUUM 的不同之处在于它访问可能包含未冻结 XID 或 MXID 的每个页面，而不仅仅是那些可能包含死元组的页面。
										# 默认为 1.5 亿笔事务。
										# 尽管用户可以将此值设置为从 0 到 20 亿之间的任何值，
										# 但 VACUUM 会默默地将有效值限制为 autovacuum_freeze_max_age 的 95%，
										# 以便定期手动 VACUUM 有机会在为表启动反环绕自动清理之前运行。
										# 有关详细信息，请参阅第 25.1.5 节。
#vacuum_freeze_min_age = 50000000
										# 指定 VACUUM 应该用来决定在扫描表时是否冻结行版本的截止年龄（在事务中）。
										# 默认为 5000 万次事务。
										# 尽管用户可以将此值设置为 0 到 10 亿之间的任何值，
										# 但 VACUUM 会默默地将有效值限制为 autovacuum_freeze_max_age 值的一半，因此强制自动清空之间的时间不会过短。
										# 有关详细信息，请参阅第 25.1.5 节。
#vacuum_failsafe_age = 1600000000
										# 指定在 VACUUM 采取特殊措施以避免系统范围的事务 ID 环绕失败之前，
										# 表的 pg_class.relfrozenxid 字段可以达到的最大年龄（在事务中）。
										# 这是 VACUUM 的最后手段。
										# 故障保护通常在用于防止事务 ID 环绕的自动清理已经运行了一段时间时触发，尽管在任何 VACUUM 期间都可能触发故障保护。

										# 触发故障保护时，将不再应用任何有效的基于成本的延迟，并且会绕过进一步的非必要维护任务（例如索引清理）。

										# 默认为 16 亿笔事务。
										# 尽管用户可以将此值设置为从 0 到 21 亿的任意值，
										# 但 VACUUM 会默默地将有效值调整为不小于 autovacuum_freeze_max_age 的 105%。
#vacuum_multixact_freeze_table_age = 150000000
										# 如果表的 pg_class.relminmxid 字段已达到此设置指定的年龄，VACUUM 将执行积极扫描。
										# 主动扫描与常规 VACUUM 的不同之处在于它访问可能包含未冻结 XID 或 MXID 的每个页面，而不仅仅是那些可能包含死元组的页面。
										# 默认值为 1.5 亿个 multixact。
										# 尽管用户可以将此值设置为 0 到 20 亿之间的任意值，
										# 但 VACUUM 会默默地将有效值限制为 autovacuum_multixact_freeze_max_age 
										# 的 95%，以便定期手动 VACUUM 有机会在为表启动反环绕之前运行。
										# 有关详细信息，请参阅第 25.1.5.1 节。
#vacuum_multixact_freeze_min_age = 5000000
										# 指定 VACUUM 应该使用的截止期限（在 multixacts 中）来决定在扫描表时是用更新的事务 ID 还是 multixact ID 
										# 替换 multixact ID。
										# 默认值为 500 万个多任务。
										# 尽管用户可以将此值设置为从 0 到 10 亿之间的任何值，
										# 但 VACUUM 会默默地将有效值限制为 autovacuum_multixact_freeze_max_age 值的一半，因此强制自动清空之间的时间不会过短。
										# 有关详细信息，请参阅第 25.1.5.1 节。
#vacuum_multixact_failsafe_age = 1600000000
										# 指定表的 pg_class.relminmxid 字段在 VACUUM 采取特殊措施以避免系统范围的 multixact ID 环绕失败之前可以达到的最大年龄
										# （在 multixact 中）。
										# 这是 VACUUM 的最后手段。故障保护通常在用于防止事务 ID 环绕的自动清理已经运行了一段时间时触发，
										# 尽管在任何 VACUUM 期间都可能触发故障保护。

										# 触发故障保护时，将不再应用任何有效的基于成本的延迟，并且会绕过进一步的非必要维护任务（例如索引清理）。

										# 默认值为 16 亿个 multixact。
										# 虽然用户可以将此值设置为从 0 到 21 亿之间的任何值，
										# 但 VACUUM 会默默地将有效值调整为不小于 autovacuum_multixact_freeze_max_age 的 105%。
#bytea_output = 'hex'                   # hex, escape
										# 设置 bytea 类型值的输出格式。
										# 有效值为十六进制（默认）和转义（传统的 PostgreSQL 格式）。
										# 有关详细信息，请参阅第 8.4 节。
										# 无论此设置如何，bytea 类型在输入时始终接受这两种格式。

#xmlbinary = 'base64'					# 设置二进制值在 XML 中的编码方式。
										# 这适用于例如当 bytea 值通过函数 xmlelement 或 xmlforest 转换为 XML 时。
										# 可能的值是 base64 和 hex，它们都在 XML Schema 标准中定义。
										# 默认值为 base64。
										# 有关 XML 相关函数的更多信息，请参阅第 9.15 节。

										# 这里的实际选择主要是个人喜好问题，仅受客户端应用程序中可能的限制的限制。
										# 这两种方法都支持所有可能的值，尽管十六进制编码会比 base64 编码稍大一些。
#xmloption = 'content'					# 设置在 XML 和字符串值之间转换时 DOCUMENT 或 CONTENT 是否是隐式的。
										# 参见第 8.13 节对此的描述。
										# 有效值为 DOCUMENT 和 CONTENT。
										# 默认值为内容。
										# 根据 SQL 标准，设置此选项的命令是
										# SET XML OPTION { DOCUMENT | CONTENT };
										# 此语法在 PostgreSQL 中也可用。
#gin_pending_list_limit = 4MB
										# 设置 GIN 索引的挂起列表的最大大小，在启用 fastupdate 时使用。
										# 如果列表增长大于此最大大小，则会通过将其中的条目批量移动到索引的主 GIN 数据结构来清理它。
										# 如果此值指定为不带单位，则以千字节为单位。
										# 默认值为四兆字节 (4MB)。
										# 通过更改索引存储参数，可以为单个 GIN 索引覆盖此设置。
										# 有关详细信息，请参阅第 67.4.1 节和第 67.5 节。

### - Locale and Formatting -

datestyle = 'iso, mdy'					# 设置日期和时间值的显示格式，以及解释模糊日期输入值的规则。
										# 出于历史原因，此变量包含两个独立组件：
										# 输出格式规范（ISO，Postgres，SQL或German）以及年/月/日序的输入/输出规范（DMY，MDY或YMD）。
										# 这些可以单独设置或一起设置。
										# 关键词Euro和European是DMY的同义词,关键词 US, NonEuro, and NonEuropean 的同义词是MDY。
										# 有关详细信息，请参见第8.5节。
										# 内置默认值为ISO，MDY，但INITDB将使用与所选LC_TIME语言环境的行为对应的设置初始化配置文件。
#intervalstyle = 'postgres'				# 设置间隔值的显示格式。
										# 值 sql_standard 将产生匹配 SQL 标准间隔文字的输出。
										# 当 DateStyle 参数设置为 ISO 时，值 postgres（这是默认值）将生成匹配 8.4 之前的 PostgreSQL 版本的输出。
										# 当 DateStyle 参数设置为非 ISO 输出时，值 postgres_verbose 将产生与 8.4 之前的 PostgreSQL 版本匹配的输出。
										# 值 iso_8601 将产生与 ISO 8601 第 4.4.3.2 节中定义的时间间隔“带指示符的格式”相匹配的输出。
										# IntervalStyle 参数也会影响模糊区间输入的解释。
										# 有关详细信息，请参阅第 8.5.4 节。
timezone = 'PRC'						# 设置显示和解释时间戳的时区。
										# 内置默认值为 GMT，但通常在 postgresql.conf 中被覆盖； 
										# initdb 将在那里安装与其系统环境相对应的设置。
										# 有关详细信息，请参阅第 8.5.3 节。
#timezone_abbreviations = 'Default'     # Select the set of available time zone
                                        # abbreviations.  Currently, there are
                                        #   Default
                                        #   Australia (historical usage)
                                        #   India
                                        # You can create your own file in
                                        # share/timezonesets/.
										# 设置服务器将接受的用于日期时间输入的时区缩写集合。
										# 默认为“Default”，这是一个适用于世界大部分地区的集合；
										# 还有“Australia”和“India”，可以为特定装置定义其他系列。
										# 有关详细信息，请参阅第 B.4 节。
#extra_float_digits = 1                 # min -15, max 3; any value >0 actually
                                        # selects precise output mode
										# 此参数调整用于浮点值文本输出的位数，包括 float4、float8 和几何数据类型。
										# 如果值为 1（默认值）或更大，则浮点值以最短精度格式输出；
										# 见第 8.1.3 节。
										# 生成的实际位数仅取决于输出的值，而不取决于此参数的值。 
										# float8 值最多需要 17 位，float4 值最多需要 9 位。
										# 这种格式既快速又精确，在正确读取时准确地保留了原始二进制浮点值。
										# 为了历史兼容性，允许值最大为 3。
										# 如果该值为零或负数，则将输出四舍五入到给定的十进制精度。
										# 使用的精度是根据此参数的值减少的类型（FLT_DIG 或 DBL_DIG，视情况而定）的标准位数。 
										# （例如，指定 -1 将导致 float4 值舍入为 5 个有效数字，float8 值舍入为 14 位。）
										# 这种格式速度较慢，不会保留二进制浮点值的所有位，但可能会更多人类可读。
#client_encoding = sql_ascii            # actually, defaults to database
                                        # encoding
										# 设置客户端编码（字符集）。默认是使用数据库编码。 PostgreSQL 服务器支持的字符集在第 24.3.1 节中描述。

#These settings are initialized by initdb, but they can be changed.
lc_messages = 'en_US.UTF-8'                     # locale for system error message
                                        # strings
										# 设置显示消息的语言。可接受的值取决于系统；有关详细信息，请参阅第 24.1 节。
										# 如果此变量设置为空字符串（这是默认值），
										# 则该值以系统相关的方式从服务器的执行环境继承。

										# 在某些系统上，此区域设置类别不存在。
										# 设置此变量仍然有效，但不会有任何效果。
										# 此外，有可能不存在所需语言的翻译消息。
										# 在这种情况下，您将继续看到英文消息。

										# 只有超级用户可以更改此设置，因为它会影响发送到服务器日志以及客户端的消息，并且不正确的值可能会掩盖服务器日志的可读性。
lc_monetary = 'en_US.UTF-8'                     # locale for monetary formatting
										# 设置用于格式化货币金额的语言环境，例如使用 to_char 系列函数。
										# 可接受的值取决于系统；有关详细信息，请参阅第 24.1 节。
										# 如果此变量设置为空字符串（这是默认值），则该值以系统相关的方式从服务器的执行环境继承。
lc_numeric = 'en_US.UTF-8'                      # locale for number formatting
										# 设置用于格式化数字的语言环境，例如使用 to_char 系列函数。
										# 可接受的值取决于系统；有关详细信息，请参阅第 24.1 节。
										# 如果此变量设置为空字符串（这是默认值），则该值以系统相关的方式从服务器的执行环境继承。
lc_time = 'en_US.UTF-8'                         # locale for time formatting
										# 设置用于格式化日期和时间的语言环境，例如使用 to_char 系列函数。
										# 可接受的值取决于系统；有关详细信息，请参阅第 24.1 节。
										# 如果此变量设置为空字符串（这是默认值），则该值以系统相关的方式从服务器的执行环境继承。

#### default configuration for text search
default_text_search_config = 'pg_catalog.english'
										# 选择文本搜索函数的变体使用的文本搜索配置，这些变体没有指定配置的显式参数。
										# 有关详细信息，请参阅第 12 章。
										# 内置默认值是 pg_catalog.simple，但如果可以识别与该区域设置匹配的配置，
										# initdb 将使用与所选 lc_ctype 区域设置对应的设置来初始化配置文件。

### - Shared Library Preloading -  共享库预加载
#有几个设置可用于将共享库预加载到服务器中，以加载附加功能或获得性能优势。
#例如，设置 '$libdir/mylib' 会导致从安装的标准库目录预加载 mylib.so（或在某些平台上为 mylib.sl）。
#设置之间的区别在于它们何时生效以及更改它们需要什么权限。
#可以通过这种方式预加载 PostgreSQL 过程语言库，通常使用语法“$libdir/plXXX”，其中 XXX 是 pgsql、perl、tcl 或 python。
#只有专门用于 PostgreSQL 的共享库才能以这种方式加载。每个 PostgreSQL 支持的库都有一个“magic block”，经过检查以保证兼容性。
#因此，不能以这种方式加载非 PostgreSQL 库。
#您也许可以为此使用诸如 LD_PRELOAD 之类的操作系统工具。
#通常，请参阅特定模块的文档以了解加载该模块的推荐方式。
#local_preload_libraries = ''
										# 此变量指定要在连接开始时预加载的一个或多个共享库。它包含一个以逗号分隔的库名称列表，其中每个名称都被解释为 LOAD 命令。
										# 条目之间的空格被忽略；如果您需要在名称中包含空格或逗号，请用双引号将库名称括起来。
										# 该参数值仅在连接开始时生效。
										# 后续更改无效。如果未找到指定的库，则连接尝试将失败。

										# 任何用户都可以设置此选项。因此，可以加载的库仅限于出现在安装标准库目录的 plugins 子目录中的库。 
										# （数据库管理员有责任确保那里只安装“安全”的库。）local_preload_libraries 中的条目可以明确指定此目录，
										# 例如 $libdir/plugins/mylib，或者只指定库名称 - mylib 将具有相同的效果为 $libdir/plugins/mylib。

										# 此功能的目的是允许非特权用户将调试或性能测量库加载到特定会话中，而无需明确的 LOAD 命令。
										# 为此，通常使用客户端上的 PGOPTIONS 环境变量或使用 ALTER ROLE SET 设置此参数。

										# 但是，除非一个模块是专门为非超级用户以这种方式使用而设计的，否则这通常不是正确的设置。
										# 而是查看 session_preload_libraries。
#session_preload_libraries = ''			# 此变量指定要在连接开始时预加载的一个或多个共享库。它包含一个以逗号分隔的库名称列表，其中每个名称都被解释为 LOAD 命令。
										# 条目之间的空格被忽略；如果您需要在名称中包含空格或逗号，请用双引号将库名称括起来。
										# 该参数值仅在连接开始时生效。后续更改无效。
										# 如果未找到指定的库，则连接尝试将失败。
										# 只有超级用户可以更改此设置。

										# 此功能的目的是允许将调试或性能测量库加载到特定会话中，而无需给出明确的 LOAD 命令。
										# 例如，通过使用 ALTER ROLE SET 设置此参数，可以为给定用户名下的所有会话启用 auto_explain。
										# 此外，无需重新启动服务器即可更改此参数（但更改仅在启动新会话时生效），因此以这种方式添加新模块更容易，
										# 即使它们应该应用于所有会话。

										# 与 shared_preload_libraries 不同，在会话开始时而不是在首次使用时加载库并没有很大的性能优势。
										# 但是，使用连接池时有一些优势。
#shared_preload_libraries = ''  # (change requires restart)
										# 此变量指定在服务器启动时要预加载的一个或多个共享库。
										# 它包含一个以逗号分隔的库名称列表，其中每个名称都被解释为 LOAD 命令。
										# 条目之间的空格被忽略；如果您需要在名称中包含空格或逗号，请用双引号将库名称括起来。
										# 此参数只能在服务器启动时设置。
										# 如果找不到指定的库，服务器将无法启动。

										# 一些库需要执行某些只能在 postmaster 启动时进行的操作，
										# 例如分配共享内存、保留轻量级锁或启动后台工作程序。
										# 这些库必须在服务器启动时通过此参数加载。有关详细信息，请参阅每个库的文档。

										# 也可以预加载其他库。通过预加载共享库，在首次使用该库时避免了库启动时间。
										# 但是，启动每个新服务器进程的时间可能会略有增加，即使该进程从不使用该库。
										# 因此，此参数仅推荐用于将在大多数会话中使用的库。
										# 此外，更改此参数需要重新启动服务器，因此这不是用于短期调试任务的正确设置，
										# 例如。使用 session_preload_libraries 代替。
#jit_provider = 'llvmjit'               # JIT library to use
										# 此变量是要使用的 JIT 提供程序库的名称（请参阅第 32.4.2 节）。
										# 默认值为 llvmjit。此参数只能在服务器启动时设置。

										# 如果设置为不存在的库，JIT 将不可用，但不会引发错误。
										# 这允许 JIT 支持与主 PostgreSQL 包分开安装。
### - Other Defaults -

#dynamic_library_path = '$libdir'		# 如果需要打开一个可动态加载的模块，并且在 CREATE FUNCTION 或 LOAD 命令中指定的文件名没有目录组件（即名称不包含斜杠），
										# 系统将在该路径中搜索所需文件。

										# dynamic_library_path 的值必须是由冒号（或 Windows 上的分号）分隔的绝对目录路径列表。
										# 如果列表元素以特殊字符串 $libdir 开头，则编译的 PostgreSQL 包库目录将替换 $libdir；
										# 这是安装标准 PostgreSQL 发行版提供的模块的地方。 （使用 pg_config --pkglibdir 找出该目录的名称。）
										# 例如：
										# dynamic_library_path = '/usr/local/lib/postgresql:/home/my_project/lib:$libdir'
										# 或者，在 Windows 环境中：
										# dynamic_library_path = 'C:\tools\postgresql;H:\my_project\lib;$libdir'
										# 此参数的默认值为“$libdir”。
										# 如果该值设置为空字符串，则会关闭自动路径搜索。

										# 超级用户可以在运行时更改此参数，但以这种方式完成的设置只会持续到客户端连接结束，因此应保留此方法用于开发目的。
										# 设置此参数的推荐方法是在 postgresql.conf 配置文件中。
#gin_fuzzy_search_limit = 0				# GIN 索引扫描返回的集合大小的软上限。有关详细信息，请参阅第 67.5 节。
										# 由于可能为每个项目插入许多键，因此插入 GIN 索引可能会很慢。
										# 因此，对于表中的批量插入，建议删除 GIN 索引并在完成批量插入后重新创建它。

										# 为 GIN 启用 fastupdate 时（请参阅第 67.4.1 节了解详细信息），惩罚小于未启用时的惩罚。
										# 但是对于非常大的更新，最好还是删除并重新创建索引。

#------------------------------------------------------------------------------
## LOCK MANAGEMENT  锁管理
#------------------------------------------------------------------------------

#deadlock_timeout = 1s					# 这是在检查是否存在死锁条件之前等待锁的时间。
										# 死锁检查相对昂贵，因此服务器不会在每次等待锁时都运行它。
										# 我们乐观地假设死锁在生产应用程序中并不常见，只需在锁上等待一段时间，然后再检查死锁。
										# 增加此值可减少不必要的死锁检查所浪费的时间，但会减慢真正死锁错误的报告速度。
										# 如果指定此值没有单位，则以毫秒为单位。
										# 默认值为一秒 (1s)，这可能是您在实践中想要的最小值。
										# 在负载很重的服务器上，您可能需要提升它。
										# 理想情况下，该设置应该超过您的典型事务时间，以提高在服务员决定检查死锁之前释放锁的几率。
										# 只有超级用户可以更改此设置。

										# 设置 log_lock_waits 时，此参数还确定在发出有关锁定等待的日志消息之前等待的时间量。
										# 如果您正在尝试调查锁定延迟，您可能希望设置比正常的 deadlock_timeout 更短的时间。
#max_locks_per_transaction = 64         # min 10
                                        # (change requires restart)
										# 共享锁表跟踪 max_locks_per_transaction * (max_connections + max_prepared_transactions) 对象（例如，表）上的锁；
										# 因此，任何时候都不能锁定这么多不同的对象。
										# 该参数控制为每个事务分配的对象锁的平均数量；
										# 只要所有事务的锁都适合锁表，单个事务就可以锁定更多对象。
										# 这不是可以锁定的行数；该值是无限的。
										# 默认值 64 在历史上已被证明是足够的，但如果您的查询在单个事务中涉及许多不同的表，
										# 例如查询具有许多子表的父表，则可能需要提高此值。
										# 此参数只能在服务器启动时设置。

										# 运行备用服务器时，您必须将此参数设置为与主服务器上相同或更高的值。
										# 否则，备用服务器中将不允许查询。
#max_pred_locks_per_transaction = 64    # min 10
                                        # (change requires restart)
										# 共享谓词锁表跟踪 max_pred_locks_per_transaction * (max_connections + max_prepared_transactions) 对象（例如，表）上的锁；
										# 因此，任何时候都不能锁定这么多不同的对象。
										# 该参数控制为每个事务分配的对象锁的平均数量；只要所有事务的锁都适合锁表，单个事务就可以锁定更多对象。
										# 这不是可以锁定的行数；该值是无限的。
										# 默认值 64 在测试中通常已足够，但如果您的客户端在单个可序列化事务中接触许多不同的表，则可能需要提高此值。
										# 此参数只能在服务器启动时设置。
#max_pred_locks_per_relation = -2       # negative values mean
                                        # (max_pred_locks_per_transaction / -max_pred_locks_per_relation) - 1
										# 这控制在锁被提升为覆盖整个关系之前，可以对单个关系的多少页或元组进行谓词锁定。
										# 大于或等于零的值表示绝对限制，而负值表示 max_pred_locks_per_transaction 除以此设置的绝对值。
										# 默认值为 -2，它保留以前版本的 PostgreSQL 的行为。
										# 此参数只能在 postgresql.conf 文件或服务器命令行中设置。
#max_pred_locks_per_page = 2            # min 0
										# 这控制在将锁提升为覆盖整个页面之前，可以对单个页面上的多少行进行谓词锁定。
										# 默认为 2。
										# 该参数只能在 postgresql.conf 文件或服务器命令行中设置。


#------------------------------------------------------------------------------
## VERSION AND PLATFORM COMPATIBILITY  版本和平台兼容性
#------------------------------------------------------------------------------

### - Previous PostgreSQL Versions -  以前的 PostgreSQL 版本

#array_nulls = on						# 这控制数组输入解析器是否将未引用的 NULL 识别为指定空数组元素。
										# 默认情况下，这是打开的，允许输入包含空值的数组值。
										# 但是，8.2 之前的 PostgreSQL 版本不支持数组中的空值，因此会将 NULL 视为指定具有字符串值“NULL”的普通数组元素。
										# 为了与需要旧行为的应用程序向后兼容，可以关闭此变量。

										# 请注意，即使此变量关闭，也可以创建包含空值的数组值。
#backslash_quote = safe_encoding        # on, off, or safe_encoding
										# 这控制引号是否可以由字符串文字中的 \' 表示。
										# 表示引号的首选 SQL 标准方法是将引号加倍 ('') 但 PostgreSQL 历来也接受 \'。
										# 但是，使用 \' 会带来安全风险，因为在某些客户端字符集编码中，存在最后一个字节在数字上等同于 ASCII \ 的多字节字符。
										# 如果客户端代码不正确地转义，则可能发生 SQL 注入攻击。
										# 通过使服务器拒绝其中引号似乎被反斜杠转义的查询，可以防止这种风险。
										# backslash_quote 的允许值为 on（始终允许 \'）、off（始终拒绝）
										# 和 safe_encoding（仅当客户端编码不允许多字节字符中的 ASCII \时才允许）。 
										# safe_encoding 是默认设置。

										# 请注意，在符合标准的字符串文字中， \只是表示 \ 无论如何。
										# 此参数仅影响对不符合标准的文字的处理，包括转义字符串语法 (E'...')。
#escape_string_warning = on
										# 打开时，如果反斜杠 (\) 出现在普通字符串文字（'...' 语法）中并且 standard_conforming_strings 关闭，则会发出警告。
										# 默认为开启。

										# 希望使用反斜杠作为转义的应用程序应修改为使用转义字符串语法 (E'...')，
										# 因为根据 SQL 标准，普通字符串的默认行为现在是将反斜杠视为普通字符。可以启用此变量来帮助定位需要更改的代码。
#lo_compat_privileges = off				# 在 9.0 之前的 PostgreSQL 版本中，大对象没有访问权限，因此，所有用户始终可以读写。
										# 将此变量设置为 on 会禁用新的权限检查，以便与以前的版本兼容。
										# 默认为关闭。
										# 只有超级用户可以更改此设置。

										# 设置此变量不会禁用所有与大对象相关的安全检查——仅禁用那些在 PostgreSQL 9.0 中更改了默认行为的安全检查。
#quote_all_identifiers = off			# 当数据库生成 SQL 时，强制所有标识符都被引用，即使它们不是（当前）关键字。
										# 这将影响 EXPLAIN 的输出以及 pg_get_viewdef 等函数的结果。
										# 另见 pg_dump 和 pg_dumpall 的 --quote-all-identifiers 选项。
#standard_conforming_strings = on		# 这控制普通字符串文字 ('...') 是否按照 SQL 标准中的规定按字面意思对待反斜杠。
										# 从 PostgreSQL 9.1 开始，默认值为 on（之前的版本默认为 off）。
										# 应用程序可以检查此参数以确定如何处理字符串文字。
										# 此参数的存在也可以被视为支持转义字符串语法 (E'...') 的指示。
										# 如果应用程序希望将反斜杠视为转义字符，则应使用转义字符串语法（第 4.1.2.2 节）。
#synchronize_seqscans = on				# 这允许大型表的顺序扫描彼此同步，以便并发扫描几乎在同一时间读取同一块，从而共享 I/O 工作负载。
										# 启用此功能后，扫描可能从表的中间开始，然后“wrap around”结束以覆盖所有行，以便与已经在进行的扫描活动同步。
										# 这可能导致没有 ORDER BY 子句的查询返回的行顺序发生不可预知的变化。
										# 将此参数设置为 off 可确保 8.3 之前的行为，即顺序扫描始终从表的开头开始。
										# 默认为开启。

### - Other Platforms and Clients - 平台和客户端兼容性

#transform_null_equals = off
										# 启用时，形式为 expr = NULL（或 NULL = expr）的表达式被视为 expr IS NULL，
										# 也就是说，如果 expr 计算结果为空值，则它们返回 true，否则返回 false。 
										# expr = NULL 正确的符合 SQL 规范的行为是始终返回 null（未知）。
										# 因此此参数默认为关闭。

										# 但是，Microsoft Access 中的过滤表单生成的查询似乎使用 expr = NULL 来测试空值，因此如果您使用该接口访问数据库，
										# 您可能需要打开此选项。由于 expr = NULL 形式的表达式总是返回 null 值（使用 SQL 标准解释），
										# 它们不是很有用，并且在正常应用程序中不经常出现，因此该选项在实践中几乎没有危害。
										# 但是新用户经常对涉及空值的表达式的语义感到困惑，所以这个选项默认是关闭的。

										# 请注意，此选项仅影响确切的形式 = NULL，而不影响其他比较运算符或其他在计算上等效于涉及等于运算符的某些表达式
										# （例如 IN）的表达式。
										# 因此，此选项不是对不良编程的一般修复。

										# 有关相关信息，请参阅第 9.2 节。

#------------------------------------------------------------------------------
## ERROR HANDLING  错误处理
#------------------------------------------------------------------------------

#exit_on_error = off                    # terminate session on any error? 
										# 如果打开，任何错误都将终止当前会话。
										# 默认情况下，它设置为关闭，因此只有 FATAL 错误才会终止会话。

#restart_after_crash = on               # reinitialize after backend crash?
										# 当设置为默认值时，PostgreSQL 将在后端崩溃后自动重新初始化。
										# 将此值设置为 on 通常是最大化数据库可用性的最佳方式。
										# 但是，在某些情况下，例如当集群件调用 PostgreSQL 时，禁用重新启动可能很有用，
										# 以便集群件可以获得控制并采取它认为合适的任何操作。

										# 此参数只能在 postgresql.conf 文件或服务器命令行中设置。
#data_sync_retry = off                  # retry or panic on failure to fsync
                                        # data?
                                        # (change requires restart)
										# 当设置为默认值 off 时，PostgreSQL 将在无法将修改的数据文件刷新到文件系统时引发 PANIC 级错误。
										# 这会导致数据库服务器崩溃。此参数只能在服务器启动时设置。

										# 在某些操作系统上，在回写失败后，内核页面缓存中的数据状态是未知的。
										# 在某些情况下，它可能已被完全遗忘，因此重试不安全；第二次尝试可能会被报告为成功，而实际上数据已经丢失。
										# 在这些情况下，避免数据丢失的唯一方法是在报告任何故障后从 WAL 中恢复，最好是在调查故障的根本原因并更换任何故障硬件之后。

										# 如果设置为 on，PostgreSQL 将改为报告错误但继续运行，以便可以在以后的检查点重试数据刷新操作。
										# 只有在调查了操作系统对缓冲数据的处理后才能将其设置为 on，以防写回失败。
#recovery_init_sync_method = fsync      # fsync, syncfs (Linux 5.8+)
										# 当设置为默认的 fsync 时，PostgreSQL 将在崩溃恢复开始之前递归地打开并同步数据目录中的所有文件。
										# 文件搜索将遵循 WAL 目录和每个配置的表空间的符号链接（但不是任何其他符号链接）。
										# 这是为了确保在重放更改之前，所有 WAL 和数据文件都持久地存储在磁盘上。
										# 这适用于启动未完全关闭的数据库集群，包括使用 pg_basebackup 创建的副本。

										# 在 Linux 上，可以使用 syncfs 来要求操作系统同步包含数据目录、WAL 文件和每个表空间的整个文件系统
										# （但不能通过符号链接访问任何其他文件系统）。
										# 这可能比 fsync 设置快很多，因为它不需要一个一个地打开每个文件。
										# 另一方面，如果文件系统由修改大量文件的其他应用程序共享，则可能会更慢，因为这些文件也将写入磁盘。
										# 此外，在 5.8 之前的 Linux 版本上，写入数据到磁盘时遇到的 I/O 错误可能不会报告给 PostgreSQL，
										# 并且相关的错误消息可能只出现在内核日志中。

										#此参数只能在 postgresql.conf 文件或服务器命令行中设置。


#------------------------------------------------------------------------------
## CONFIG FILE INCLUDES
#------------------------------------------------------------------------------

#These options allow settings to be loaded from files other than the
#default postgresql.conf.  Note that these are directives, not variable
#assignments, so they can usefully be given more than once.
#PostgreSQL 提供了一些功能来将复杂的 postgresql.conf 文件分解为子文件。
#在管理具有相关但不相同的配置的多台服务器时，这些功能特别有用。

#除了单独的参数设置之外，postgresql.conf 文件还可以包含包含指令，这些指令指定另一个要读取和处理的文件，就好像它此时已插入到配置文件中一样。
#此功能允许将配置文件划分为物理上独立的部分。
#包含指令看起来像：
#include 'filename'
#如果文件名不是绝对路径，则将其视为相对于包含引用配置文件的目录。包含可以嵌套。

#还有一个 include_if_exists 指令，其作用与 include 指令相同，除非引用的文件不存在或无法读取。
#常规包含将认为这是一个错误情况，但 include_if_exists 仅记录一条消息并继续处理引用配置文件。

#postgresql.conf 文件还可以包含 include_dir 指令，这些指令指定要包含的配置文件的整个目录。这些看起来像
#include 'filename'
#如果文件名不是绝对路径，则将其视为相对于包含引用配置文件的目录。包含可以嵌套。

#还有一个 include_if_exists 指令，其作用与 include 指令相同，除非引用的文件不存在或无法读取。
#常规包含将认为这是一个错误情况，但 include_if_exists 仅记录一条消息并继续处理引用配置文件。


#postgresql.conf 文件还可以包含 include_dir 指令，这些指令指定要包含的配置文件的整个目录。这些看起来像
#include_dir 'directory'
#非绝对目录名称被视为相对于包含引用配置文件的目录。在指定目录中，仅包含名称以 .conf 结尾的非目录文件。
#以 . 开头的文件名。字符也被忽略，以防止错误，因为此类文件在某些​​平台上是隐藏的。
#包含目录中的多个文件按文件名顺序处理（根据 C 语言环境规则，即数字在字母之前，大写字母在小写字母之前）。

#包含文件或目录可用于在逻辑上分隔数据库配置的各个部分，而不是使用单个大的 postgresql.conf 文件。
#考虑一家拥有两台数据库服务器的公司，每台服务器都有不同的内存量。
#对于诸如日志记录之类的事情，配置的一些元素可能会共享。
#但是服务器上与内存相关的参数会在两者之间有所不同。并且可能还有特定于服务器的自定义。
#管理这种情况的一种方法是将站点的自定义配置更改分成三个文件。您可以将其添加到 postgresql.conf 文件的末尾以包含它们：
#include 'shared.conf'
#include 'memory.conf'
#include 'server.conf'

#所有系统都将具有相同的 shared.conf。
#每个具有特定内存量的服务器可以共享相同的 memory.conf；
#您可能有一个用于所有具有 8GB RAM 的服务器，另一个用于具有 16GB 内存的服务器。
#最后 server.conf 可以在其中包含真正特定于服务器的配置信息。

#另一种可能性是创建一个配置文件目录并将此信息放入那里的文件中。例如，可以在 postgresql.conf 的末尾引用一个 conf.d 目录：
#include_dir 'conf.d'
#然后你可以像这样命名 conf.d 目录中的文件：
#00shared.conf
#01memory.conf
#02server.conf
#此命名约定建立了加载这些文件的明确顺序。这很重要，因为只有在服务器读取配置文件时遇到的特定参数的最后一个设置才会被使用。
#在此示例中，conf.d/02server.conf 中设置的内容将覆盖 conf.d/01memory.conf 中设置的值。

#您可以改为使用这种方法来描述性地命名文件：
#00shared.conf
#01memory-8GB.conf
#02server-foo.conf

#这种安排为每个配置文件变体提供了一个唯一的名称。当多台服务器的配置都存储在一个位置（例如版本控制存储库中）时，
#这有助于消除歧义。 （在版本控制下存储数据库配置文件是另一个值得考虑的好习惯。）

#include_dir 'directory'

#include_dir = '...'                    # include files ending in '.conf' from
                                        # a directory, e.g., 'conf.d'
#include_if_exists = '...'              # include file only if it exists
#include = '...'                        # include file


#------------------------------------------------------------------------------
## CUSTOMIZED OPTIONS
#------------------------------------------------------------------------------

#Add settings for extensions here
#shared_preload_libraries = 'ptrack, pgaudit, auto_explain, pg_start_sql, pg_stat_statements, pg_stat_kcache, pg_crash'
shared_preload_libraries = 'ptrack, pgaudit, auto_explain, pg_start_sql, pg_stat_statements, pg_stat_kcache'
ptrack.map_size=64
#crash.signals = '1 2 3'
#crash.delay = 30
auto_explain.log_min_duration = '3s'

log_destination = 'csvlog'
logging_collector = on
log_connections = on
```