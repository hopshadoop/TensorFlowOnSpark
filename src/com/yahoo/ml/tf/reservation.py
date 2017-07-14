# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.
from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function

import logging
import pickle
import select
import socket
import struct
import threading
import time

BUFSIZE = 1024

class Reservations:

  def __init__(self, required):
    self.required = required
    self.lock = threading.RLock()
    self.reservations = []

  def add(self, meta):
    with self.lock:
      self.reservations.append(meta)

  def done(self):
    with self.lock:
      return len(self.reservations) >= self.required

  def get(self):
    with self.lock:
      return self.reservations

  def remaining(self):
    with self.lock:
      return self.required - len(self.reservations)

class GPUPresence:
  def __init__(self, required):
    self.required = required
    self.lock = threading.RLock()
    self.gpus_presence = []

  def add(self, meta):
    with self.lock:
        self.gpus_presence.append(meta)

  def done(self):
    with self.lock:
        return len(self.gpus_presence) >= self.required

  def get(self):
    with self.lock:
        return self.gpus_presence

  def remaining(self):
    with self.lock:
        return self.required - len(self.gpus_presence)

class MessageSocket(object):
  """Abstract class w/ length-prefixed socket send/receive functions"""

  def receive(self, sock):
    msg = None
    data = b''
    recv_done = False
    recv_len = -1
    while not recv_done:
      buf = sock.recv(BUFSIZE)
      if buf == None or len(buf) == 0:
        raise Exception("socket closed")
      if recv_len == -1:
        recv_len = struct.unpack('>I', buf[:4])[0]
        data += buf[4:]
        recv_len -= len(data)
      else:
        data += buf
        recv_len -= len(buf)
      recv_done = (recv_len == 0)

    msg = pickle.loads(data)
    return msg

  def send(self, sock, msg):
    data = pickle.dumps(msg)
    buf = struct.pack('>I', len(data)) + data
    sock.sendall(buf)

class Server(MessageSocket):
  """Simple socket server with length prefixed pickle messages"""
  reservations = None
  done = False
  gpu_presence = None
  gpu_done = False

  def __init__(self, count):
    assert count > 0
    self.reservations = Reservations(count)
    self.gpu_presence = GPUPresence(count)

  def gpus_are_present():
    for executor in cluster_info:
      if executor['gpu_present'] == True:
        return True
    return False

  def await_reservations(self):
    """Block until all reservations done"""
    while not self.reservations.done():
      logging.info("waiting for {0} reservations".format(self.reservations.remaining()))
      time.sleep(1)
    logging.info("all reservations completed")
    #If GPU(s) have been requested for workers
    if gpus_are_present():
        logging.info("mapping executors with GPUs to be workers, and rest to be parameter servers")
        switch_all_wrongly_placed_ps()
    return self.reservations.get()

  #Modifies clusterspec to switch executor for parameter server and worker
  #The places are switched only if any parameter server have a GPU, meaning atleast one worker does not have a GPU
  def switch_all_wrongly_placed_ps():
    cluster_info = self.reservations.get()

    for executor in cluster_info:
          #This is not allowed, one of the workers should be run on this executor
          #We need to fix it!
          if executor['job_name'] == 'ps' and executor['gpu_present']:
            ps_worker_num = executor['worker_num']
            ps_task_index = executor['task_index']

            #Found a worker with no GPU, but all should have GPUs!
            for candidate_replacement in cluster_info:
              if candidate_replacement['job_name'] == 'worker' and candidate_replacement['gpu_present'] == False:
                executor['job_name'] = 'worker'
                executor['worker_num'] = candidate_replacement['worker_num']
                executor['task_index'] = candidate_replacement['task_index']
                candidate_replacement['job_name'] = 'ps'
                candidate_replacement['worker_num'] = ps_worker_num
                candidate_replacement['task_index'] = ps_task_index
                break

  def handle_message(self, sock, msg):
    logging.debug("received: {0}".format(msg))
    msg_type = msg['type']
    if msg_type == 'REG':
      self.reservations.add(msg['data'])
      MessageSocket.send(self, sock, 'OK')
    if msg_type == 'REG_EXECUTOR_GPU_PRESENCE':
      self.gpu_presence.add(msg['data'])
      MessageSocket.send(self, sock, 'OK')
    elif msg_type == 'QUERY':
      MessageSocket.send(self, sock, self.reservations.done())
    elif msg_type == 'GPU_QUERY':
      MessageSocket.send(self, sock, self.gpu_presence.done())
    elif msg_type == 'QINFO':
      rinfo = self.reservations.get()
      MessageSocket.send(self, sock, rinfo)
    elif msg_type == 'GPU_INFO':
      rinfo = self.gpu_presence.get()
      MessageSocket.send(self, sock, rinfo)
    elif msg_type == 'STOP':
      logging.info("setting server.done")
      MessageSocket.send(self, sock, 'OK')
      self.done = True
    else:
      MessageSocket.send(self, sock, 'ERR')

  def start(self):
    """Start listener in a background thread"""
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(('',0))
    server_sock.listen(1)

    host = socket.gethostname()
    port = server_sock.getsockname()[1]
    addr = (host,port)
    logging.info("listening for reservations at {0}".format(addr))

    def _listen(self, sock):
      CONNECTIONS = []
      CONNECTIONS.append(sock)

      while not self.done:
        read_socks, write_socks, err_socks = select.select(CONNECTIONS, [], [], 60)
        for sock in read_socks:
          if sock == server_sock:
            client_sock, client_addr = sock.accept()
            CONNECTIONS.append(client_sock)
            logging.debug("client connected from {0}".format(client_addr))
          else:
            try:
              msg = self.receive(sock)
              self.handle_message(sock, msg)
            except Exception as e:
              logging.debug(e)
              sock.close()
              CONNECTIONS.remove(sock)

      server_sock.close()

    t = threading.Thread(target=_listen, args=(self, server_sock))
    t.daemon = True
    t.start()

    return addr

  def stop(self):
    """Stop the socket listener"""
    self.done = True

class Client(MessageSocket):
  """Client to register/await node reservations"""
  sock = None

  def __init__(self, server_addr):
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.sock.connect(server_addr)
    logging.info("connected to server at {0}".format(server_addr))

  def _request(self, msg_type, msg_data=None):
    """Helper function to wrap msg w/ msg_type"""
    msg = {}
    msg['type'] = msg_type
    if msg_data:
      msg['data'] = msg_data
    MessageSocket.send(self, self.sock, msg)
    logging.debug("sent: {0}".format(msg))
    resp = MessageSocket.receive(self, self.sock)
    logging.debug("received: {0}".format(resp))
    return resp

  def close(self):
    self.sock.close()

  def register(self, reservation):
    """Register reservation with server"""
    resp = self._request('REG', reservation)
    return resp

  def get_reservations(self):
    """Get current list of reservations"""
    cluster_info = self._request('QINFO')
    return cluster_info

  def await_reservations(self):
    """Poll until all reservations completed, then return cluster_info"""
    done = False
    while not done:
      done = self._request('QUERY')
      time.sleep(1)
    return self.get_reservations()

  def register_gpu_presence(self, gpu_is_present):
    """Register gpu_presence with server"""
    resp = self._request('REG_EXECUTOR_GPU_PRESENCE', gpu_is_present)
    return resp

  def get_gpu_presence(self):
    """Get current list of reservations"""
    gpu_present = self._request('GPU_INFO')
    return gpu_present

  def await_gpu_check(self):
    """Poll until all checks have been done to find GPUs are completed, then return True if any GPUs are present"""
    done = False
    while not done:
      done = self._request('GPU_QUERY')
      time.sleep(1)
    gpu_presence_arr = self.get_gpu_presence()
    for gpu_present in gpu_presence_arr:
        if gpu_present:
            return True
    return False

  def request_stop(self):
    resp = self._request('STOP')
    return resp

