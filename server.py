#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import shutil
from utillc import *
import cherrypy
import time
import os
import torch


fileDir = os.path.dirname(os.path.abspath(__file__))

config = {
  '/' : {
      'tools.staticdir.on': True,
      'tools.staticdir.dir': os.path.join(fileDir, 'www'),
    },
  'global' : {
      'server.socket_host' : '0.0.0.0', #192.168.1.5', #'127.0.0.1',
    'server.socket_port' : 8080,
    'server.thread_pool' : 8,
  }
}

html = """
<!DOCTYPE html>
<html>
  <head>
    <title>torch test</title>
  </head>
  <body>
    CONTENT
  </body>
</html>
"""

class App:

    def __init__(self) :
        EKO()
        
    @cherrypy.expose
    def index(self):
        try :
            avail = torch.cuda.is_available()
            a = torch.tensor(1)
            cap = torch.cuda.get_device_capability()
            res = "cuda available"  + str(avail) + '\n'
            res = "capability " + str(cap) + '\n'
        except Exception as e:
            res = str(e)
        
        return html.replace("CONTENT", res)


if __name__ == '__main__':
    app = App()
    cherrypy.log.error_log.propagate = False
    cherrypy.log.access_log.propagate = False
    cherrypy.quickstart(app, '/', config)
  
