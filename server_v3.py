import socket
import time
import threading
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Description of your program')

    parser.add_argument('-host', type=str, default='localhost', help='Server host')
    parser.add_argument('-port', type=int, default=12345, help='Port for TCP')
    parser.add_argument('-nclients', type=int, default=1, help='Maximum client numbers')

    args = parser.parse_args()

    return args


class WebcamServer:
    def __init__(self, host='localhost', port=12345, nclients=1):
        self.host = host
        self.port = port
        self.nclients = nclients
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_sockets = []
        self.client_addresses = []

        self.clients_states = []
        for i in range(nclients):
            self.clients_states.append('notready')

        self.start_flag = False
        self.film_flag = False

        self.start_server()

    def handle_client(self, socket, address, index):
        try:
            # sending order
            socket.sendall("ready".encode())

            while True:
                # receive order from client
                data = socket.recv(1024).decode()
                if data == 'ready':  # client is ready
                    print('Client {}({}) is ready'.format(index, address))
                    self.clients_states[index] = 'ready'
                elif data == "start":
                    print('Client {}({}) starts to display.'.format(index, address))
                    self.clients_states[index] = 'start'
                elif data == 'end':
                    print('Client {}({}) ends for displaying, turn to "ready" stage.'.format(index, address))
                    self.clients_states[index] = 'ready'
                elif data == "film":
                    print('Client {}({}) starts to recording.'.format(index, address))
                    self.clients_states[index] = 'film'
                elif data == 'stop':
                    print('Client {}({}) ends for recording, turn to "start" stage.'.format(index, address))
                    self.clients_states[index] = 'start'
                elif data == "exit":
                    print('Client {}({}) exit.'.format(index, address))
                    self.clients_states[index] = 'exit'
                elif data.startswith("show"):
                    case_idx = data.split(";")[1]
                    times_idx = int(data.split(";")[2])
                    print("===================================================================")
                    print(f"In case {case_idx}, and in {times_idx} times")
                    print("===================================================================")
                    
                else:
                    print('Error: Client {}({}) sent a exception: {}, connection close.'.format(index, address, data))
                    self.clients_states[index] = 'disconnected'
                    break

        except (ConnectionResetError, ConnectionAbortedError):
            print('Client {}({}) disconnected.'.format(index, address))
            self.clients_states[index] = 'disconnected'

    def send_command(self, command):
        for client_socket in self.client_sockets:
            client_socket.sendall(command.encode())

    def check_states(self, command):
        while True:  # Ensure all clients is ready
            for idx, state in enumerate(self.clients_states):
                if state == 'disconnected':
                    continue
                if state != command:
                    break
            else:
                print('All clients is in "{}" state.'.format(command))
                break

    def check_connection(self):
        connected = 0
        for client_state in self.clients_states:
            if client_state != 'disconnected':
                connected += 1

        if connected < self.nclients:
            return False
        else:
            return True

    def start_server(self):
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(self.nclients)
        print("Waiting for clients to connect...")
        print(f"Client {len(self.client_sockets)}/{self.nclients} is connected.")

        client_index = 0
        while len(self.client_sockets) < self.nclients:
            client_socket, client_address = self.server_socket.accept()
            print(f"Client from {client_address} is connected.")
            print(f"Client {len(self.client_sockets)}/{self.nclients} is connected.")

            self.client_sockets.append(client_socket)
            self.client_addresses.append(client_address)

            client_thread = threading.Thread(target=self.handle_client, args=(client_socket, client_address, client_index))
            client_thread.start()
            client_index += 1

        self.check_states('ready')
        print(f"All clients is connected, Ready for data collection")

        # sending order in main thread
        while True:
            connection_state = self.check_connection()
            if not connection_state:
                print('Disconnected client is detected, close all the connections')  # Need to write as update table
                if self.film_flag:
                    self.send_command('stop')
                    self.check_states('start')
                    self.film_flag = False
                if self.start_flag:
                    self.send_command('end')
                    self.check_states('ready')
                    self.start_flag = False
                self.send_command('exit')
                self.check_states('exit')
                break

            command = input("Insert command (start / film / stop / end) or type 'exit' to quit: ")
            if command in ['start', 'end', 'film', 'stop', 'exit']:
                if command == 'start':
                    if self.start_flag:
                        print("All clients have started, skip.")
                        continue
                    self.send_command('start')
                    self.check_states('start')
                    self.start_flag = True
                    print(f"All clients start, ready for recording")
                elif command == 'end':
                    if not self.start_flag:
                        print("All clients do not start, skip.")
                        continue
                    if self.film_flag:
                        print("All clients remains recording, command 'stop' needs to be performed first.")
                        continue
                    self.send_command('end')
                    self.check_states('ready')
                    self.start_flag = False
                elif command == 'film':
                    cnt = 0
                    while True:
                        if not self.start_flag:
                            print("All clients do not start, command 'start' needs to be performed first.")
                            continue
                        self.send_command('film')
                        self.check_states('film')
                        self.film_flag = True
                        time.sleep(10)
                        self.send_command('stop')
                        self.check_states('start')
                        self.film_flag = False

                        cnt += 1

                        if cnt >= 10:
                            break

                elif command == 'stop':
                    if not self.film_flag:
                        print('All clients do not record, skip.')
                        continue
                    self.send_command('stop')
                    self.check_states('start')
                    self.film_flag = False
                elif command == 'exit':
                    if self.film_flag:
                        print('All clients are still recording, command "stop" needs to be performed first.')
                        continue
                    if self.start_flag:
                        print('All clients are still displaying, command "end" needs to be performed first.')
                        continue
                    self.send_command('exit')
                    self.check_states('exit')
                    break
            else:
                print('Invalid command: {}'.format(command))

        for client_socket in self.client_sockets:
            client_socket.close()

        self.server_socket.close()


if __name__ == "__main__":
    args = get_args()

    server = WebcamServer(
        host=args.host,
        port=args.port,
        nclients=args.nclients
    )
