# -*- coding: UTF-8 -*-
import socket
import asyncio

async def echo_server(addr):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(addr)
    sock.listen(5)
    sock.setblocking(False)
    while True:
        client, addr = await loop.sock_accept(sock)
        print("connect from ", addr)
        loop.create_task(echo_handler(client))

async def echo_handler(client):
    with client:
        while True:
            try:
                data = await loop.sock_recv(client, 10000)
                if not data:
                    break
                await loop.sock_sendall(client, str.encode("Got: ") + data)
            except Exception as e:
                client.close()
                print("connection closed")
                break
    print("connection closed")


loop = asyncio.get_event_loop()
loop.create_task(echo_server(('0.0.0.0', 25000)))
loop.run_forever()