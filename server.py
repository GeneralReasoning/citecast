from openreward.environments import Server

from citecast import CiteCast

if __name__ == "__main__":
    server = Server([CiteCast])
    server.run()
