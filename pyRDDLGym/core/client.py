import socket
import xml.etree.ElementTree as xmltree

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.policy import BaseAgent


EOM = '\n\n\n'


class RDDLSimClient:
    """Creates a TCP/IP client that listens to the provided port and passes
    messages between a pyRDDLGym BaseAgent and a server that is
    designed to interact with rddlsim (https://github.com/ssanner/rddlsim)."""

    def __init__(self, policy: BaseAgent, port: int=2323):
        self.policy = policy
        self.address = ("127.0.0.1", port)
    
    def run(self):

        print("INFO: Establishing socket...", flush=True)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with sock:
            # Force the connection to this port (sometimes it stays locked after repeated runs).
            # https://stackoverflow.com/questions/4465959/python-errno-98-address-already-in-use
            print("INFO: Forcing connection...", flush=True)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # connect to the server
            sock.connect(self.address)
            print(f"INFO: Connected to server at address {self.address[0]} "
                  f"with port {self.address[1]}.\n", flush=True)
            
            # signal to the server that ready to start the session
            self.send_message(sock, self.build_session_request())

            # receive the session request message
            message = self.receive_message(sock)
            self.process_session_msg(message)
            print("INFO: Session initialized.\n", flush=True)

            # repeat over rounds...
            currentround = 0
            while True:
                currentround += 1

                # send the round request to the server
                print(f"INFO: Starting round {currentround}...", flush=True)
                self.send_message(sock, self.build_round_request())
                message = self.receive_message(sock)
                rounds_left = self.process_round_msg(message)

                # process the initial state of the round
                message = self.receive_message(sock)
                state = self.process_state_msg(message)

                # continue over decision epochs...
                while True:

                    # send the action from the policy to the server
                    actions = self.policy.sample_action(state)
                    self.send_message(sock, self.build_action_request(actions))
                    
                    # receive the next state from the server
                    message = self.receive_message(sock)
                    state = self.process_state_msg(message)   
                    if state is None:
                        break    
                
                if rounds_left <= 0:
                    break
            
        sock.close()
        print("INFO: Socket closed.\n", flush=True)

    def send_message(self, connection, msg):
        connection.sendall(str.encode(msg + EOM))

    def build_session_request(self):
        msg = '<session-request>'
        msg += '<problem-name>domain</problem-name>'
        msg += '<client-name>client</client-name>'
        msg += '<instance-name>instance</instance-name>'
        msg += '<input-language>rddl</input-language>'
        msg += '</session-request>'
        return msg
    
    def build_round_request(self):
        msg = '<round-request>'
        msg += '<execute-policy>yes</execute-policy>'
        msg += '</round-request>'
        return msg
    
    def build_action_request(self, actions):
        msg = '<actions>'
        for key in actions:
            msg = msg + "<action>"
            fluent_name = key.split(RDDLPlanningModel.FLUENT_SEP)[0]
            objects = key.split(RDDLPlanningModel.FLUENT_SEP)[1:]
            objects = RDDLPlanningModel.FLUENT_SEP.join(objects)
            objects = objects.split(RDDLPlanningModel.OBJECT_SEP)
            msg = msg + "<action-name>" + fluent_name + "</action-name>"
            for object in objects:
                msg = msg + "<action-arg>" + object + "</action-arg>"
            msg = msg + "<action-value>" + str(actions[key]).lower() + "</action-value>"
            msg = msg + "</action>"
        msg += "</actions>"
        return msg
    
    def receive_message(self, connection):
        message = ''
        while not message.endswith(EOM):
            data = connection.recv(8192)
            if data:
                data = data.decode("UTF-8")
                message += data
            else:
                print("FATAL: connection lost.")
                exit(1)
        message = message.split(EOM)[0]
        return message
    
    def process_session_msg(self, data):
        parser = xmltree.XMLParser()
        root = xmltree.fromstring(data, parser)
        if root.tag != "session-init":
            print("ERROR: Malformed session init message: "
                  "session-init tag missing.", flush=True)
            exit(1)
        
    def process_round_msg(self, data):
        parser = xmltree.XMLParser()
        root = xmltree.fromstring(data, parser)
        if root.tag != "round-init":
            print("ERROR: Malformed round init message: "
                  "round-init tag missing.", flush=True)
            exit(1)
        roundsleft = int(root.find("rounds-left").text.strip())
        return roundsleft
    
    def process_state_msg(self, data):
        parser = xmltree.XMLParser()
        root = xmltree.fromstring(data, parser)
        if root.tag == 'round-end':
            return None
        elif root.tag != "turn":
            print("ERROR: Malformed turn message: turn tag missing.", flush=True)
            exit(1)
        states = root.findall('observed-fluent')
        result = {}
        for state in states:
            name = state.find("fluent-name").text
            args = state.findall("fluent-arg")
            separator = RDDLPlanningModel.FLUENT_SEP
            for arg in args:
                name = name + separator + arg.text
                separator = RDDLPlanningModel.OBJECT_SEP
            value = state.find("fluent-value").text
            result[name] = value
        return result
