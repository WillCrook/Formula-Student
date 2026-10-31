"""Isolated disposable server for DOM workflow checks; not the user's workspace."""
from pathlib import Path
import json
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from werkzeug.serving import make_server, WSGIRequestHandler
from viewer import create_app
from workspace import ROOT

class QuietHandler(WSGIRequestHandler):
    def log_request(self,*args,**kwargs):
        pass

if __name__=="__main__":
    with tempfile.TemporaryDirectory(dir=ROOT/"tmp",prefix="ui-check-") as temporary:
        server=make_server("127.0.0.1",0,create_app(Path(temporary),update_report=False),request_handler=QuietHandler)
        print(json.dumps({"url":f"http://127.0.0.1:{server.server_port}"}),flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
