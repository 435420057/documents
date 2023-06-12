FROM ubuntu:20.04
MAINTAINER examples@docker.com

RUN apt-get update && apt-get install -y openssh-server supervisor
RUN mkdir -p /var/run/sshd /var/log/supervisor /data /etc/supervisor/conf.d

#COPY supervisord.conf /etc/supervisor/supervisord.conf

EXPOSE 22 80 8001 4567 8088 8080 8081 8082 8083 8084 8085 8086 8087 8089 9999 1002
CMD ["/usr/bin/supervisord", "-n", "-c","/etc/supervisor/supervisord.conf"]