import base64
import json
import socket
import xml.etree.ElementTree as xmltree

from pyRDDLGym import RDDLEnv
from pyRDDLGym.Core.Compiler.RDDLModel import PlanningModel


class RDDLSimAgent:
    """creates a TCP/IP server that listens to the provided port and passes
    messages between a pyRDDLGym environment and a client that is
    designed to interact with rddlsim (https://github.com/ssanner/rddlsim)"""

    def __init__(self, domain, instance, numrounds, time, port=2323):
        # concatenate domain and instance files
        print(f"loading domain {domain}...", flush=True)
        f = open(domain)
        self.task = f.read()
        f.close()
        print(f"loading instance {instance}...", flush=True)
        f = open(instance)
        self.task = self.task + f.read()
        f.close()

        # encode task
        print(f"encoding task for sharing in TCP connections...", flush=True)
        self.task = base64.b64encode(str.encode(self.task))
        self.task = self.task.decode("ascii")

        # create RDDLEnv
        print(f"creating RDDL environment...", flush=True)
        self.env = RDDLEnv.RDDLEnv(domain=domain, instance=instance)

        # initialize RDDLSimAgent
        self.roundsleft = numrounds
        self.currentround = 0
        self.time = time
        self.address = ("127.0.0.1", port)
        self.client = ""
        self.problem = ""
        self.total_reward = 0.0

        # Data in case there is a dump request
        self.logs = []

    def run(self):
        """starts the RDDLSimAgent to wait for a planner to connect"""

        print(f"establishing socket...", flush=True)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        with sock:
            # Force the connection to this port (sometimes it stays locked after repeated runs).
            # https://stackoverflow.com/questions/4465959/python-errno-98-address-already-in-use
            print(f"forcing connection...", flush=True)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            sock.bind(self.address)
            print(
                f"listening at address {self.address[0]} with port {self.address[1]}...",
                flush=True,
            )
            sock.listen(1)
            connection, client_address = sock.accept()
            with connection:
                self.run_session(connection)
            connection.close()
        sock.close()
        self.env.close()

    def run_session(self, connection):
        """runs an interactive session between the pyRDDLGym environment
        and a connected rddlsim client and terminates afterwards"""

        # handle session request
        data = self.receive_message(connection)
        self.process_init_session_request(data)
        print(f"session request from {self.client} for {self.problem}", flush=True)
        msg = self.build_session_request_msg()
        self.send_message(connection, msg)
        session_request_expected = False
        print("session initialized", flush=True)

        while self.roundsleft > 0:
            self.run_round(connection)

        msg = self.build_session_end_msg()
        self.send_message(connection, msg)

    def run_round(self, connection):
        self.logs.append([])

        # handle round request
        data = self.receive_message(connection)
        self.process_round_request(data)
        print(f"starting round {self.currentround}", flush=True)
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

            self.logs[-1].append(
                {
                    "state": json.loads(
                        str(state)
                        .replace("'", '"')
                        .replace("True", "true")
                        .replace("False", "false")
                    ),
                    "actions": json.loads(
                        str(actions).replace("'", '"').replace('"true"', "true")
                    ),
                }
            )

            next_state, reward, _, _, _ = self.env.step(actions)

            self.logs[-1][-1]["reward"] = float(reward)

            round_reward += reward
            state = next_state

            turn = turn + 1
            if turn == self.env.horizon:
                msg = self.build_round_end_msg(reward, round_reward)
                self.send_message(connection, msg)
                self.total_reward += round_reward

                self.logs[-1].append(
                    {
                        "state": json.loads(
                            str(state)
                            .replace("'", '"')
                            .replace("True", "true")
                            .replace("False", "false")
                        ),
                        "actions": False,
                        "round_reward": float(round_reward),
                    }
                )

                break

            msg = self.build_state_msg(state, turn, reward)
            self.send_message(connection, msg)

    def dump_data(self, fn):
        """Dumps the data to a json file"""
        with open(fn, "w") as f:
            json.dump(self.logs, f)

    def send_message(self, connection, msg):
        # print(f"sending message: {msg}")
        connection.send(str.encode(msg))

    def receive_message(self, connection):
        data = connection.recv(8192)
        if data:
            data = data.decode("UTF-8")
            data = data[:-1]
            # print(f"received message: {data}")
        else:
            print("Error: connection lost", flush=True)
            exit(1)
        return data

    def build_session_request_msg(self):
        msg = "<session-init>"
        msg = msg + "<task>" + str(self.task) + "</task>"
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
        # print(state)
        msg = "<turn>"
        msg = msg + "<turn-num>" + str(turn) + "</turn-num>"
        msg = msg + "<time-left>1000</time-left>"
        msg = msg + "<immediate-reward>" + str(rew) + "</immediate-reward>"
        for key in state:
            msg = msg + "<observed-fluent>"
            fluent_name = key.split(PlanningModel.FLUENT_SEP)[0]
            objects = key.split(PlanningModel.FLUENT_SEP)[1:]
            objects = PlanningModel.FLUENT_SEP.join(objects)
            objects = objects.split(PlanningModel.OBJECT_SEP)
            var = key.split("_")
            msg = msg + "<fluent-name>" + fluent_name + "</fluent-name>"
            for object in objects:
                msg = msg + "<fluent-arg>" + object + "</fluent-arg>"
            msg = msg + "<fluent-value>" + str(state[key]).lower() + "</fluent-value>"
            msg = msg + "</observed-fluent>"
        msg = msg + "</turn>"
        return msg

    def build_round_end_msg(self, rew, round_reward):
        msg = "<round-end>"
        msg = msg + "<instance-name>" + self.problem + "</instance-name>"
        msg = msg + "<client-name>" + self.client + "</client-name>"
        msg = msg + "<round-num>" + str(self.currentround) + "</round-num>"
        msg = msg + "<round-reward>" + str(round_reward) + "</round-reward>"
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
        if root.tag != "session-request":
            print(
                "Malformed session request message: session-request tag missing",
                flush=True,
            )
            exit(1)
        self.problem = root.find("problem-name").text
        self.client = root.find("client-name").text
        input_language = root.find("input-language").text
        if input_language != "rddl":
            print(
                "Malformed session request message: input language must be rddl",
                flush=True,
            )
            exit(1)

    def process_round_request(self, data):
        parser = xmltree.XMLParser()
        root = xmltree.fromstring(data, parser)
        if root.tag != "round-request":
            print(
                "Malformed round request message: round-request tag missing", flush=True
            )
            exit(1)
        execute = root.find("execute-policy").text.strip()
        if execute != "yes":
            print(
                "Malformed round request message: policy must be executed", flush=True
            )
            exit(1)
        self.currentround += 1
        self.roundsleft -= 1

    def process_action(self, data):
        parser = xmltree.XMLParser()
        root = xmltree.fromstring(data, parser)
        if root.tag != "actions":
            print("Malformed action message: actions tag missing", flush=True)
            exit(1)
        actions = root.findall("action")
        result = {}
        for act in actions:
            name = act.find("action-name").text
            args = act.findall("action-arg")
            separator = PlanningModel.FLUENT_SEP
            for arg in args:
                name = name + separator + arg.text
                separator = PlanningModel.OBJECT_SEP
            value = act.find("action-value").text
            result[name] = value
        return result
