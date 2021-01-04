# -*- coding: UTF-8 -*-

def parrot(voltage, state='a stiff', action='voom'):
     print("-- This parrot wouldn't", action, end=' ')
     print("if you put", voltage, "volts through it.", end=' ')
     print("E's", state, "!", end=' [ Y ]')


d = {"voltage": "four million", "state": "bleedin' demised", "action": "VOOM"}
parrot(**d)

bt = bytearray(b'123456')
print(dict(one=b'eee', two=2, three=3), bt)