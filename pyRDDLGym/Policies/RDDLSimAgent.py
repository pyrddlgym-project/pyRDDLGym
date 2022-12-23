import base64
import os
import socket
import xml.etree.ElementTree as xmltree

from pyRDDLGym import RDDLEnv

class RDDLSimAgent:
    ''' creates a TCP/IP server that listens to the provided port and passes
    messages between a pyRDDLGym environment and a client that is
    designed to interact with rddlsim (https://github.com/ssanner/rddlsim)'''
    
    def __init__(self, domain, instance, numrounds, time, port=2323):
        self.env = RDDLEnv.RDDLEnv(domain=domain, instance=instance)

        # read domain file
        f = open(domain)
        self.domain = f.read()
        self.domain = os.linesep.join([s for s in self.domain.splitlines() if s.strip()])
        self.domain = base64.b64encode(str.encode(self.domain))
        self.domain = self.domain.decode("ascii")
        f.close()

        # read instance file
        f = open(instance)
        self.instance = f.read()
        self.instance = os.linesep.join([s for s in self.instance.splitlines() if s.strip()])
        self.instance = base64.b64encode(str.encode(self.instance))
        self.instance = self.instance.decode("ascii")
        f.close()

        # initialize RDDLSimAgent
        self.roundsleft = numrounds
        self.currentround = 0
        self.time = time
        self.address = ("127.0.0.1", port)
        self.client = ""
        self.problem = ""
        self.total_reward = 0.0

    def run(self):
        ''' starts the RDDLSimAgent to wait for a planner to connect'''
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with sock:
            sock.bind(self.address)
            sock.listen(1)
            connection, client_address = sock.accept()
            with connection:
                self.run_session(connection)
            connection.close()
        sock.close()
        self.env.close()

    def run_session(self, connection):
        ''' runs an interactive session between the pyRDDLGym environment
        and a connected rddlsim client and terminates afterwards'''
        
        # handle session request
        data = self.receive_message(connection)
        self.process_init_session_request(data)
        print(f"session request from {self.client} for {self.problem}")
        msg = self.build_session_request_msg()
        self.send_message(connection, msg)
        session_request_expected = False
        print("session initialized")
                        
        while self.roundsleft > 0:
            self.run_round(connection)

        msg = self.build_session_end_msg()
        self.send_message(connection, msg)

    def run_round(self, connection):
        # handle round request
        data = self.receive_message(connection)
        self.process_round_request(data)
        print(f"starting round {self.currentround}")
        msg = self.build_round_request_msg()
        self.send_message(connection, msg)

        # initialize round
        state = self.env.reset()
        round_reward = 0.0
        turn = 1
        msg = self.build_state_msg(state, turn, 0.0)
        self.send_message(connection, msg)

        # run round
        while True:
            data = self.receive_message(connection)
            actions = self.process_action(data)
            next_state, reward, done, info = self.env.step(actions)
            round_reward += reward
            state = next_state
    
            turn = turn + 1
            if turn == self.env.horizon:
                msg = self.build_round_end_msg(reward, round_reward)
                self.send_message(connection, msg)
                self.total_reward += round_reward
                break
            else:
                msg = self.build_state_msg(state, turn, 0.0)
                self.send_message(connection, msg)

    def send_message(self, connection, msg):
        #print(f"sending message: {msg}")
        connection.send(str.encode(msg))

    def receive_message(self, connection):
        data = connection.recv(8192)
        if data:
            data = data.decode('UTF-8')
            data = data[:-1]
            #print(f"received message: {data}")
        else:
            print("Error: connection lost")
            exit(1)
        return data                        
                            
    def build_session_request_msg(self):
        msg = "<session-init>"
        msg = msg + "<task>" + str(self.domain) + str(self.instance) + "</task>"
        msg = msg + "<session-id>0</session-id>"
        msg = msg + "<num-rounds>" + str(self.roundsleft) + "</num-rounds>"
        msg = msg + "<time-allowed>" + str(self.time) + "</time-allowed>"
        msg = msg + "</session-init>"
        return msg

    def build_round_request_msg(self):
        msg = "<round-init>"
        msg = msg + "<round-num>" + str(self.currentround) + "</round-num>"
        msg = msg + "<time-left>1000</time-left>"
        msg = msg + "<rounds-left>" + str(self.roundsleft) + "</rounds-left>"
        msg = msg + "<sessionID>0</sessionID>"
        msg = msg + "</round-init>"
        return msg

    def build_state_msg(self, state, turn, rew):
        #print(state)
        msg = "<turn>"
        msg = msg + "<turn-num>" + str(turn) + "</turn-num>"
        msg = msg + "<time-left>1000</time-left>"
        msg = msg + "<immediate-reward>" + str(rew) + "</immediate-reward>"
        for key in state:
            msg = msg + "<observed-fluent>"
            var = key.split("_")
            msg = msg + "<fluent-name>" + var[0] + "</fluent-name>"
            var = var[1:]
            for param in var:
                msg = msg + "<fluent-arg>" + param + "</fluent-arg>"
            msg = msg + "<fluent-value>" + str(state[key]).lower() + "</fluent-value>"            
            msg = msg + "</observed-fluent>"
        msg = msg + "</turn>"
        return msg

    def build_round_end_msg(self, rew, round_reward):
        msg = "<round-end>"
        msg = msg + "<instance-name>" + self.problem + "</instance-name>"
        msg = msg + "<client-name>" + self.client + "</client-name>"
        msg = msg + "<round-num>" + str(self.currentround) + "</round-num>"
        msg = msg + "<round-reward>" +  str(round_reward) + "</round-reward>"
        msg = msg + "<turns-used>" + str(self.env.horizon) + "</turns-used>"
        msg = msg + "<time-left>1000</time-left>"
        msg = msg + "<immediate-reward>" + str(rew) + "</immediate-reward>"
        msg = msg + "</round-end>"
        return msg

    def build_session_end_msg(self):
        msg = "<session-end>"
        msg = msg + "<instance-name>" + self.problem + "</instance-name>"
        msg = msg + "<total-reward>" + str(self.total_reward) + "</total-reward>"
        msg = msg + "<rounds-used>" + str(self.currentround) + "</rounds-used>"
        msg = msg + "<time-used>0</time-used>"
        msg = msg + "<client-name>" + self.client + "</client-name>"
        msg = msg + "<session-id>0</session-id>"
        msg = msg + "<time-left>1000</time-left>"
        msg = msg + "</session-end>"
        return msg

    def process_init_session_request(self, data):
        parser = xmltree.XMLParser()
        root = xmltree.fromstring(data, parser)
        if (root.tag != "session-request"):
            print("Malformed session request message: session-request tag missing")
            exit(1)
        self.problem = root.find("problem-name").text
        self.client = root.find("client-name").text
        input_language = root.find("input-language").text
        if (input_language != "rddl"):
            print("Malformed session request message: input language must be rddl")
            exit(1)

    def process_round_request(self, data):
        parser = xmltree.XMLParser()
        root = xmltree.fromstring(data, parser)
        if (root.tag != "round-request"):
            print("Malformed round request message: round-request tag missing")
            exit(1)
        execute = root.find("execute-policy").text.strip()
        if execute != "yes":
            print("Malformed round request message: policy must be executed")
            exit(1)
        self.currentround += 1
        self.roundsleft -= 1

    def process_action(self, data):
        parser = xmltree.XMLParser()
        root = xmltree.fromstring(data, parser)
        if (root.tag != "actions"):
            print("Malformed action message: actions tag missing")
            exit(1)
        actions = root.findall("action")
        result = {}
        for act in actions:
            name = act.find("action-name").text
            args = act.findall("action-arg")
            for arg in args:
                name = name + "_" + arg.text
            value = act.find("action-value").text
            result[name] = value
        return result
