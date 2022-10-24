#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import shutil
from utillc import *
import cherrypy
import time
import os
import torch
import numpy

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

        class ES :
            def __init__(self) :
                self.r = ""
            def write(self, s) :
                self.r += "<br> \n" + s + "</br>"
            def flush(self) :
                pass
        es = ES()
        utillc.ekostream = es
        EKOT("CHERRY SERVER STATUS")
        try :
            import torch, torchvision; 
            EKOX(torch.__version__)
            EKOX(torch.version.cuda)
            EKOX(torchvision.__version__)
            a=torch.rand(5,3).cuda()
            EKOX(torch.cuda.get_device_properties(0))
            EKOT('so far so good')
        except Exception as e :
            EKOX(str(e))
        EKOX(numpy.__version__)
        try :
            import pytorch3d
            EKOX(pytorch3d.__version__)
            EKOT('all good!')   
        except Exception as e :
            EKOX(e);
        EKOT('next tensorboard..')
        try :
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter('/tmp/runs')
            print('tensorboard ok')
        except :
            print(' pb avec tensorboard')

        try :
            import torch_geometric
            EKOX(torch_geometric.__version__)
        except Exception as e :
            EKOX(e)
        try :
            EKOT("jax")
            import jax.numpy as jnp
            from jax import jit
            from jax.lib import xla_bridge  
            EKOX(xla_bridge.get_backend().platform)
            def slow_f(x):
                # Element-wise ops see a large benefit from fusion
                y = x * x + x * 2.0
                return y
            x = jnp.ones((5000, 5000))
            fast_f = jit(slow_f)
            EKO()
            # ~ 4.5 ms / loop on Titan X
            [fast_f(x) for i in range(10)]
            EKOT('fast')
            # ~ 14.5 ms / loop (also on GPU via JAX) 
            [ slow_f(x) for i in range(10)]
            EKOT('slow')
        except Exception as e :
            EKOX(e)
        EKOT("other modules..")
        import importlib
        modules = [ 'skimage', 'networkx', 'PyQt5', 'trimesh', 'cv2', 'imageio', 'matplotlib']
        for m in modules : 
            EKON(m)
            mm = importlib.import_module(m)
            try :
               EKOX(mm.__version__)
            except :
               pass
        EKOX('success')

        res = es.r
        try :
            avail = torch.cuda.is_available()
            a = torch.tensor(1)
            cap = torch.cuda.get_device_capability()

            
            res += "cuda available"  + str(avail) + '\n'
            res += "capability " + str(cap) + '\n'
            
        except Exception as e:
            res = str(e)
        
        return html.replace("CONTENT", res)


if __name__ == '__main__':
    app = App()
    app.index()
    cherrypy.log.error_log.propagate = False
    cherrypy.log.access_log.propagate = False
    cherrypy.quickstart(app, '/', config)
  
